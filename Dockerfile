FROM nvcr.io/nvidia/pytorch:24.05-py3

WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

ENTRYPOINT [ "bash" ]