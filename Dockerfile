FROM python:3.10

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./artifacts/vit_lrp2.pth /code/artifacts/vit_lrp2.pth
COPY ./models/ /code/models/
COPY ./baselines/ /code/baselines/
COPY ./inference/vit_worker.py /code/main.py

CMD ["python", "main.py"]
