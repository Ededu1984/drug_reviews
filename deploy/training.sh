aws sagemaker create-training-job \
    --training-job-name training-new \
    --algorithm-specification TrainingImage=614823179766.dkr.ecr.us-east-1.amazonaws.com/imagem-teste:latest,TrainingInputMode=File \
    --role-arn arn:aws:iam::614823179766:role/service-role/AmazonSageMaker-ExecutionRole-20230613T005694 \
    --input-data-config '[
    {
      "ChannelName": "training",
      "DataSource": {
        "S3DataSource": {
          "S3DataType": "S3Prefix",
          "S3Uri": "s3://mlops-teste/input/",
          "S3DataDistributionType": "FullyReplicated"
        }
      },
      "ContentType": "application/x-recordio",
      "CompressionType": "None",
      "RecordWrapperType": "None"
    }
  ]' \
    --output-data-config S3OutputPath=s3://mlops-teste/output/ \
    --resource-config '{
    "InstanceCount": 1,
    "InstanceType": "ml.m4.xlarge",
    "VolumeSizeInGB": 50
  }' \
    --stopping-condition '{
    "MaxRuntimeInSeconds": 86400
  }' \
    