#aws ecr create-repository --repository-name drug-reviews
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 614823179766.dkr.ecr.us-east-1.amazonaws.com/drug-reviews
docker tag drug-reviews:latest 614823179766.dkr.ecr.us-east-1.amazonaws.com/drug-reviews
docker push 614823179766.dkr.ecr.us-east-1.amazonaws.com/drug-reviews