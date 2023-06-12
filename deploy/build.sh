#aws ecr create-repository --repository-name training-teste
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 614823179766.dkr.ecr.us-east-1.amazonaws.com/training-teste
docker tag training-teste:latest 614823179766.dkr.ecr.us-east-1.amazonaws.com/training-teste
docker push 614823179766.dkr.ecr.us-east-1.amazonaws.com/training-teste