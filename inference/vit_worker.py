import pika
import json
import os
from models.LRPViT import LRPViT
from PIL import Image
from io import BytesIO
import numpy as np
import base64
from segmentation.Segmentation import FaceSeg
import cv2

def Normalize(SegMask, LRPImage):
    target_size = (224, 224) 
    if LRPImage.shape[0:2] != target_size:
        LRPImage = cv2.resize(LRPImage, target_size)
    if SegMask.shape[0:2] != target_size:
        SegMask = cv2.resize(SegMask, target_size)

    if LRPImage.dtype != np.uint8:
        LRPImage = LRPImage.astype(np.uint8)
    if SegMask.dtype != np.uint8:
        SegMask = SegMask.astype(np.uint8)
    
    return SegMask, LRPImage

def MergeSegLRP(SegMasks, LRPImage):
    LRPImage = LRPImage[:, :, 2]
    PIXEL_COUNT = {x:[] for x in SegMasks}
    for i in SegMasks:
        SegMask = SegMasks[i]
        SegMask, LRPImage = Normalize(SegMask, LRPImage)
        mask_lrp_classwise = cv2.bitwise_and(SegMask, LRPImage)
        intensity_sum = np.sum(mask_lrp_classwise)
        PIXEL_COUNT[i].append(intensity_sum)
    return PIXEL_COUNT

def photo_response(img):
    cam = model.get_cam(img)
    formatted = (cam * 255 / np.max(cam)).astype("uint8")
    lrp = Image.fromarray(formatted)

    buffered = BytesIO()
    lrp.save(buffered, format="JPEG")
    lrp_str = base64.b64encode(buffered.getvalue())

    response = {
        "faces":[
            {
            "probability": float(model.predict(img)[1]) * 100,
            "lrp": lrp_str.decode()
            }
        ]
    }
    return response


def video_response(vid):
    with open('temp.mp4', 'wb') as vid_file:
        vid_file.write(base64.b64decode(vid))



def on_request(ch, method, props, body):
    request = json.loads(body)
    resp = {"error": "Invalid request"}
    if "img" in request:
        img = Image.open(BytesIO(base64.b64decode(request["img"])))
        resp = photo_response(img)
    if "vid" in request:
        vid = request["vid"]
        resp = video_response(vid)

    ch.basic_publish(
        exchange="",
        routing_key=props.reply_to,
        properties=pika.BasicProperties(correlation_id=props.correlation_id),
        body=json.dumps(resp),
    )
    ch.basic_ack(delivery_tag=method.delivery_tag)


model = LRPViT()
video_model = 

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
