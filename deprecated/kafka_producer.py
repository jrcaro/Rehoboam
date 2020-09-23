import logging
from kafka import KafkaProducer
from utils import scrapingCameras
from PIL import Image
import io
import time

logging.basicConfig(level=logging.INFO)
count = 0

INPUT_TOPIC = 'tensorflow'
BOOTSTRAP_SERVERS= 'localhost:9094'
producer = KafkaProducer(bootstrap_servers=BOOTSTRAP_SERVERS)
"""Creates a producer to send the values to predict"""

for i in range(5):
  image = scrapingCameras([17], './', False)
  time.sleep(5)
  producer.send(INPUT_TOPIC, image)
  """ Sends the value to predict to Kafka"""
  producer.flush()