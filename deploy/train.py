
import os
import joblib
#import credentials
import argparse
import boto3
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from transformers import AutoTokenizer
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Configurar o nível de registro
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

import warnings
warnings.filterwarnings('ignore')

logging.info('Setting up the boto client!!')
#3_client = boto3.client('s3', aws_access_key_id=credentials.aws_access_key_id, aws_secret_access_key=credentials.aws_secret_access_key)
s3_client = boto3.client('s3')

def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf

if __name__ == "__main__":
    print("extracting arguments")
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    parser.add_argument("--model_dir", type=str, default="/opt/ml/")
    parser.add_argument("--s3_model_dir", type=str, default="output/")
    parser.add_argument("--key_train", type=str, default="input/drugsComTrain_processed.csv")
    parser.add_argument("--key_test", type=str, default="input/drugsComTest_processed.csv")
    parser.add_argument("--bucket", type=str, default="mlops-teste")
    
    args, _ = parser.parse_known_args()

    # Definir o nome do bucket S3 e o caminho de destino para o modelo
    logging.info('Reading the files from S3!!')

    bucket_name = args.bucket
    key_train = args.key_train
    key_test = args.key_test
    obj_train = s3_client.get_object(Bucket=bucket_name, Key=key_train)
    obj_test= s3_client.get_object(Bucket=bucket_name, Key=key_test)

    train_processed = pd.read_csv(obj_train['Body'])
    test_processed = pd.read_csv(obj_test['Body'])

    train_processed = train_processed[pd.notna(train_processed['review'])]
    test_processed = test_processed[pd.notna(test_processed['review'])]

    sentences = list(train_processed['review'])

    train_filtered = train_processed[(train_processed['condition'] == 'Birth Control') | 
                                    (train_processed['condition'] == 'Depression') |
                                    (train_processed['condition'] == 'Pain') |
                                    (train_processed['condition'] == 'Anxiety') |
                                    (train_processed['condition'] == 'Acne')]

    test_filtered = test_processed[(test_processed['condition'] == 'Birth Control') | 
                                (test_processed['condition'] == 'Depression') |
                                (test_processed['condition'] == 'Pain') |
                                (test_processed['condition'] == 'Anxiety') |
                                (test_processed['condition'] == 'Acne')]

    condition_dict = {}

    for n, i in enumerate(train_filtered.condition.unique()):
        condition_dict[i] = n
        
    train_filtered.condition = train_filtered.condition.map(condition_dict)
    test_filtered.condition = test_filtered.condition.map(condition_dict)

    sentences = list(train_filtered['review'])

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    train_filtered['tokens'] = train_filtered['review'].apply(lambda x: tokenizer(x)['input_ids'])
    test_filtered['tokens'] = test_filtered['review'].apply(lambda x: tokenizer(x)['input_ids'])

    epochs = 2
    num_classes = 5
    max_length = 10
    vocab_size = 30522
    embedding_dim = 768

    all_tokens = [i for i in train_filtered.tokens]
    all_tokens_test = [i for i in test_filtered.tokens]

    class MyModel(keras.Model):
        def __init__(self, vocab_size, embedding_dim, num_classes, max_length):
            super(MyModel, self).__init__()
            self.embed = Embedding(vocab_size, embedding_dim, input_length=max_length)
            self.lstm = LSTM(64, return_sequences=False)
            self.dense = Dense(num_classes, activation='softmax')

        def call(self, input_tensor):
            x = self.embed(input_tensor)
            x = self.lstm(x)
            x = self.dense(x)
            return x
    
    X_train = pad_sequences(all_tokens, maxlen=max_length)
    y_train = np.array(train_filtered.condition).reshape((-1, 1))
    X_test = pad_sequences(all_tokens_test, maxlen=max_length)
    y_test = np.array(test_filtered.condition).reshape((-1, 1))

    # Batch_size precisa ser múltiplo do número de observações do conjunto de treinamento
    model = MyModel(vocab_size=vocab_size,
                    embedding_dim=embedding_dim, 
                    num_classes=num_classes, 
                    max_length=max_length)

    batch_size = 32

    # Creating datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

    optimizer = keras.optimizers.Adam() 
    loss_fn = keras.losses.SparseCategoricalCrossentropy()

    logging.info('Starting the training!!')

    # Loop de treinamento
    for epoch in range(epochs):
        # Inicializar as métricas
        train_loss = keras.metrics.Mean()
        train_accuracy = keras.metrics.SparseCategoricalAccuracy()

        # Loop através dos lotes de treinamento
        for x_batch_train, y_batch_train in train_dataset:
            # Calcular os gradientes dentro de um contexto de gravação de gradientes
            with tf.GradientTape() as tape:
                # Obter as previsões do modelo
                y_pred = model(x_batch_train, training=True)
                # Calcular a perda
                loss_value = loss_fn(y_batch_train, y_pred)

            # Calcular gradientes
            grads = tape.gradient(loss_value, model.trainable_variables)
            # Atualizar os pesos do modelo
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Atualizar as métricas
            train_loss(loss_value)
            train_accuracy(y_batch_train, y_pred)

        # Imprimir métricas de treinamento ao final de cada época
        print(f"Epoch {epoch + 1}: Loss = {train_loss.result()}, Accuracy = {train_accuracy.result()}")

    # Avaliar o modelo em um conjunto de teste (opcional)
    test_loss = keras.metrics.Mean()
    test_accuracy = keras.metrics.SparseCategoricalAccuracy()
    for x_batch_test, y_batch_test in test_dataset:
        y_pred = model(x_batch_test, training=False)
        loss_value = loss_fn(y_batch_test, y_pred)
        test_loss(loss_value)
        test_accuracy(y_batch_test, y_pred)

    print(f"Test Loss: {test_loss.result()}, Test Accuracy: {test_accuracy.result()}")

    model.compile(
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
        optimizer = keras.optimizers.Adam(),
        metrics = ["accuracy"]
    )

    # Inicializar as métricas
    test_loss = 0.0
    test_accuracy = 0.0
    num_batches = 0

    logging.info('Evaluation!!')
    # Loop sobre o conjunto de teste
    for x_batch, y_batch in test_dataset:
        # Calcular as métricas para o lote atual
        loss, accuracy = model.evaluate(x_batch, y_batch)
        
        # Acumular as métricas
        test_loss += loss
        test_accuracy += accuracy
        num_batches += 1

    # Calcular as métricas médias
    test_loss /= num_batches
    test_accuracy /= num_batches

    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)

    previsoes = model.predict(test_dataset, batch_size=batch_size, verbose=2)

    # Aplicando um limiar de 0.5 para obter valores binários
    class_predictions = np.argmax(previsoes, axis=1)

    from sklearn.metrics import classification_report

    # Gerando o relatório de classificação
    report = classification_report(y_test, class_predictions)
    print(report)

    # Salvar o modelo em um arquivo pickle
    #with open(args.model_dir, 'wb') as arquivo:
    #   pickle.dump(model, arquivo)
        
    # persist model
    path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, path)
        
    # Fazer o upload do arquivo do modelo para o S3
    #logging.info('Saving the file in the bucket!!')
    #s3_client.upload_file(args.model_dir+"train.py", args.bucket, args.s3_model_dir+"model.joblib")

