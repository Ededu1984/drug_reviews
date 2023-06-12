import boto3
import urllib3
import credentials

# Desativa os avisos de SSL (não recomendado em produção)
urllib3.disable_warnings()

def create_task_definition(image_uri, client):
    response = client.register_task_definition(
        family='training-task',
        taskRoleArn='arn:aws:iam::614823179766:role/ecs-teste',
        containerDefinitions=[
            {
                'name': 'training-container',
                'image': image_uri,
                'memory': 2048,
                'cpu': 1024,
                'essential': True,
                'environment': [
                    {
                        'name': 'ENV_VARIABLE_1',
                        'value': 'VALUE_1'
                    },
                    {
                        'name': 'ENV_VARIABLE_2',
                        'value': 'VALUE_2'
                    }
                ],
                'command': ['python', 'train.py'],
                'logConfiguration': {
                    'logDriver': 'awslogs',
                    'options': {
                        'awslogs-group': '/ecs/training-logs',
                        'awslogs-region': 'us-east-1',
                        'awslogs-stream-prefix': 'training'
                    }
                }
            }
        ]
    )
    
    return response['taskDefinition']['taskDefinitionArn']

def create_task(cluster, task_definition, client):
    #networkConfiguration = {"awsvpcConfiguration":{"subnets":["subnet-0250ce53e344a8637"], 'assignPublicIp': 'ENABLED'}}  
    response = client.run_task(
        cluster=cluster,
        taskDefinition=task_definition,
        launchType='EC2',
        #networkConfiguration=networkConfiguration
    )
    return response

# Exemplo de uso
if __name__ == '__main__':
    client = boto3.client('ecs', aws_access_key_id = credentials.aws_access_key_id, aws_secret_access_key=credentials.aws_secret_access_key, verify=False)
    image_uri = '614823179766.dkr.ecr.us-east-1.amazonaws.com/training-teste:latest'
    cluster = 'arn:aws:ecs:us-east-1:879976175372:cluster/AWSBatch-dev-circulabi-batch-ce-5566ff52-0b16-3c31-a365-00691ed00fe3'
    task_definition = create_task_definition(image_uri, client)
    response = create_task(cluster, task_definition, client)
    #print(response)