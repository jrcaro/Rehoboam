import tensorflow as tf
import logging
from kafka import KafkaConsumer
from utils import detect
from PIL import Image
import io
import time
from tensorflow.python.saved_model import tag_constants
import numpy as np
import json

config = {
  'weights': './data/models/YOLO/yolov4.weights',
  'input_size': 416,
  'score_thres': 0.8,
  'model': 'yolov4',
  'weights_tf': './checkpoints/yolov4-416',
  'output_path': 'result.jpg',
  'iou': 0.45
}

saved_model_loaded = tf.saved_model.load(config['weights_tf'], tags=[tag_constants.SERVING])
infer = saved_model_loaded.signatures['serving_default']

logging.basicConfig(level=logging.INFO)

INPUT_TOPIC = 'tensorflow'
output_topic = 'predict_result'
BOOTSTRAP_SERVERS= 'localhost:9094'

consumer = KafkaConsumer(INPUT_TOPIC, bootstrap_servers=BOOTSTRAP_SERVERS, group_id="tensor_group")
"""Creates a consumer to receive the predictions"""

for i,msg in enumerate(consumer):
    t = time.time()
    img = io.BytesIO(msg.value)
    img_pil = Image.open(img)
    img_cv2 = np.array(img_pil)

    scores, classes = detect(parameters=config, infer_model=infer, image=img_cv2)
    elapse = time.time() - t
    '''result = {
        'scores': scores,
        'classes': classes,
        'time': elapse
    }

    response_to_kafka = json.dumps(result).encode()

    producer.send(output_topic, response_to_kafka)  
    producer.flush()'''

    consumer.commit()
