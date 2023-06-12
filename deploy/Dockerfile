FROM python:3.10-slim

RUN mkdir /opt/ml
WORKDIR /opt/ml

RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    apt-get install -y awscli

RUN pip install s3fs boto3 pandas scikit-learn tensorflow transformers

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml:${PATH}"

COPY train.py /opt/ml
COPY credentials.py /opt/ml

CMD ["python", "/opt/ml/train.py"]

