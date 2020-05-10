FROM python:3.8-buster

WORKDIR /app/py

RUN apt update && \
    apt install libsndfile1 -y && \
    pip install poetry && \
    poetry config virtualenvs.create false && \
    pip install jupyterlab
