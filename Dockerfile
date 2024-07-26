FROM python:3.10

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./artifacts/vit_deepfakes_large.pth /code/artifacts/vit_deepfakes_large.pth
COPY ./models/DeepViT.py /code/models/DeepViT.py
COPY ./inference/vit_worker.py /code/main.py

CMD ["python", "main.py"]
