import pika
import json
import os
from models.DeepViT import DeepViT
from PIL import Image
from io import BytesIO
import base64


def on_request(ch, method, props, body):
    request = json.loads(body)
    print(f"Got {request}")

    img = Image.open(BytesIO(base64.b64decode(request["img"])))
    response = {"probability": float(model.predict_image(img, True)[0])}

    ch.basic_publish(
        exchange="",
        routing_key=props.reply_to,
        properties=pika.BasicProperties(correlation_id=props.correlation_id),
        body=json.dumps(response),
    )
    ch.basic_ack(delivery_tag=method.delivery_tag)


model = DeepViT()

HOST = os.environ["RABBITMQ_HOST"]
USERNAME = os.environ["RABBITMQ_USERNAME"]
PASSWORD = os.environ["RABBITMQ_PASSWORD"]

credentials = pika.PlainCredentials(USERNAME, PASSWORD)
connection = pika.BlockingConnection(
    pika.ConnectionParameters(host=HOST, port=5672, credentials=credentials)
)

channel = connection.channel()
channel.queue_declare(queue="rpc_queue")
channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue="rpc_queue", on_message_callback=on_request)

print(" [x] Awaiting RPC requests")
channel.start_consuming()
