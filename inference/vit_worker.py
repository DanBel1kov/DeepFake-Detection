import pika
import json
import os
from models.LRPViT import LRPViT
from PIL import Image
from io import BytesIO
import numpy as np
import base64


def on_request(ch, method, props, body):
    request = json.loads(body)
    print(f"Got {request}")

    img = Image.open(BytesIO(base64.b64decode(request["img"])))
    cam = model.get_cam(img)
    formatted = (cam * 255 / np.max(cam)).astype("uint8")
    lrp = Image.fromarray(formatted)

    buffered = BytesIO()
    lrp.save(buffered, format="JPEG")
    lrp_str = base64.b64encode(buffered.getvalue())

    response = {
        "probability": float(model.predict(img)[1]) * 100,
        "lrp": lrp_str.decode(),
    }
    ch.basic_publish(
        exchange="",
        routing_key=props.reply_to,
        properties=pika.BasicProperties(correlation_id=props.correlation_id),
        body=json.dumps(response),
    )
    ch.basic_ack(delivery_tag=method.delivery_tag)


model = LRPViT()

HOST = os.environ["RABBITMQ_HOST"]
USERNAME = os.environ["RABBITMQ_DEFAULT_USER"]
PASSWORD = os.environ["RABBITMQ_DEFAULT_PASS"]

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
