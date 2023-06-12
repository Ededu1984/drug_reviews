aws sagemaker create-training-job \
    --training-job-name training-teste11 \
    --algorithm-specification TrainingImage=614823179766.dkr.ecr.us-east-1.amazonaws.com/training-teste:latest,TrainingInputMode=File \
    --role-arn arn:aws:iam::614823179766:role/sagemaker \
    --input-data-config '[
    {
      "ChannelName": "train",
      "DataSource": {
        "S3DataSource": {
          "S3DataType": "S3Prefix",
          "S3Uri": "s3://mlops-teste/input",
          "S3DataDistributionType": "FullyReplicated"
        }
      },
      "ContentType": "application/x-recordio",
      "CompressionType": "None",
      "RecordWrapperType": "None"
    }
  ]' \
    --output-data-config S3OutputPath=s3://mlops-teste/output \
    --resource-config '{
    "InstanceCount": 1,
    "InstanceType": "ml.m4.xlarge",
    "VolumeSizeInGB": 50
  }' \
    --stopping-condition '{
    "MaxRuntimeInSeconds": 3600
  }'

    