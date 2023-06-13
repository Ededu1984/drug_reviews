#aws ecr create-repository --repository-name imagem-teste
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 614823179766.dkr.ecr.us-east-1.amazonaws.com/imagem-teste
docker tag imagem-teste:latest 614823179766.dkr.ecr.us-east-1.amazonaws.com/imagem-teste
docker push 614823179766.dkr.ecr.us-east-1.amazonaws.com/imagem-teste