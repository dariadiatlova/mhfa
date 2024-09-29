FROM nvcr.io/nvidia/pytorch:23.09-py3

WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

ENTRYPOINT [ "bash" ]