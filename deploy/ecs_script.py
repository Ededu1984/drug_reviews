import boto3
import variables

ECS_CLUSTER_NAME = variables.ECS_CLUSTER_NAME
DOCKER_IMAGE_URI = variables.DOCKER_IMAGE_URI
ECS_CONTAINER_NAME = variables.ECS_CONTAINER_NAME
ECS_TASK_DEFINITION_NAME = variables.ECS_TASK_DEFINITION_NAME
ECS_TASK_ROLE_ARN = variables.ECS_TASK_ROLE_ARN
ECS_EXECUTION_ROLE_ARN = variables.ECS_EXECUTION_ROLE_ARN
SUBNETES = variables.SUBNETES
SECURITY_GROUP = variables.SECURITY_GROUP

local_execution = True

if local_execution:
    import credentials
    client = boto3.client('ecs', aws_access_key_id=credentials.aws_access_key_id, aws_secret_access_key=credentials.aws_secret_access_key)
else:
    client = boto3.client('ecs')

def create_cluster():
    try:
        response = client.create_cluster(
            clusterName=ECS_CLUSTER_NAME,
            settings=[
                {
                    'name': 'containerInsights',
                    'value': 'enabled'
                },
            ],
            configuration={
                'executeCommandConfiguration': {
                    'logging': 'OVERRIDE',
                    'logConfiguration': {
                        'cloudWatchLogGroupName': 'ecs',
                        'cloudWatchEncryptionEnabled': False,
                        's3BucketName': 'drug-reviews',
                        's3EncryptionEnabled': True,
                        's3KeyPrefix': 'logs/'
                    }
                }
            },
            capacityProviders=[
                'FARGATE',
            ],
            defaultCapacityProviderStrategy=[
                {
                    'capacityProvider': 'FARGATE',
                    'weight': 1,
                    'base': 1
                },
            ],
        )
        if response['ResponseMetadata']['HTTPStatusCode'] == 200:
            return True
        else:
            return False
    except Exception as e:
        print(str(e))
        return False
    
create_cluster()
print('Created ECS Cluster')

def create_task_definition():
    try:
        response = client.register_task_definition(
                family='drug-reviews-family',
                taskRoleArn=ECS_TASK_ROLE_ARN,
                executionRoleArn=ECS_EXECUTION_ROLE_ARN,
                containerDefinitions=[
                    {
                        "name": "drug-reviews-container",
                        "image": ECS_CONTAINER_NAME,
                        "cpu": 256,
                        "memory": 512,
                        "essential": True,
                        "entryPoint": ['python', 'train.py'],
                        "logConfiguration": {
                            "logDriver": "awslogs",
                            "options": {
                                "awslogs-group": "ecs",
                                "awslogs-region": "us-east-1",
                                "awslogs-stream-prefix": "drug-reviews"
                            }
                        }
                    }
                ],
                networkMode="awsvpc",
                requiresCompatibilities= [
                    "FARGATE"
                ],
                cpu= "512",
                memory= "1024")
        return response['taskDefinition']
    except Exception as e:
        print(str(e))
        return False

task_definition_family = create_task_definition()
print('Created task definition')
   
def run_task():    
    try:
        response = client.run_task(
            cluster=ECS_CLUSTER_NAME,
            taskDefinition=task_definition_family['taskDefinitionArn'],
            count=1,
            launchType='FARGATE',
            networkConfiguration={
                'awsvpcConfiguration': {
                    'subnets': SUBNETES,
                    'securityGroups': SECURITY_GROUP,
                    'assignPublicIp': 'DISABLED'
                }
            }
        )
        if response['ResponseMetadata']['HTTPStatusCode'] == 200:
            return response['tasks']
        else:
            return None
    except Exception as e:
        print(str(e))
        return None

run_task()
print('Run task')