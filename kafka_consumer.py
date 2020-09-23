from kafka import KafkaConsumer
from PIL import Image
import cv2
import io
import avro.schema
from avro.io import DatumReader
from tensorflow.python.saved_model import tag_constants
import tensorflow as tf
import logging
from utils import detect, mongoConnect
from datetime import datetime
import numpy as np
from utils import decode

config = {
    'weights': './data/models/YOLO/yolov4-obj_6000.weights',
    'input_size': 416,
    'score_thres': 0.8,
    'model': 'yolov4',
    'weights_tf': './data/models/YOLO/yolov4_imbalanced.weights',
    'output_path': 'data/result.jpg',
    'iou': 0.45
}

def main():
    #Path of the classes file
    path_names = 'data/models/YOLO/obj.names'
    #Path of the Avro scheme
    path_avro = "data/scheme.avsc"

    #Read the class file and transform in dictionary
    with open(path_names) as f:
        names_dict = {i: line.split('\n')[0] for i,line in enumerate(f)}

    #Connection to the Mongo collection to write in
    mongo_collection = mongoConnect()

    #Load the tf model
    saved_model_loaded = tf.saved_model.load(
        config['weights_tf'], tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    #logging.basicConfig(level=logging.INFO)

    #Read the Avro schema and create the Kafka consumer
    schema = avro.schema.Parse(open(path_avro, "r").read())
    reader = DatumReader(schema)
    consumer = KafkaConsumer('input_image', bootstrap_servers=['localhost:9094', 'localhost:9095'],
                        group_id="rehoboam", value_deserializer=lambda m: decode(m, reader))

    for msg in consumer:
        #Inicialize a dictionary with the class and the number of apperence to 0
        default_document = {i: 0 for i in sorted(names_dict.values(), key=lambda x: x)}
        
        #Transform the bytes to image
        img = io.BytesIO(msg.value['image'])
        img_pil = Image.open(img)
        img_cv2 = np.array(img_pil)
        
        #Call the model to detect the classes
        scores, classes = detect(
            parameters=config, infer_model=infer, image=img_cv2)

        real_classes = [i for i,j in zip(classes[0], scores[0]) if j > 0.0]
        
        unique, count = np.unique(real_classes, return_counts=True)
        result_dict = {names_dict[i]: int(j) for i,j in zip(unique, count)}

        #Store the number of incidences in the default dictionary
        for k,v in result_dict.items():
            default_document[k] = v

        result = {
          'district_id': int(msg.value['district']),
          'camera_id': int(msg.value['camera']),
          'timestamp': datetime.now(),
          'results': default_document
        }

        #Save it in Mongo
        mongo_collection.insert_one(result).inserted_id  
        consumer.commit()

if __name__ == "__main__":
    main()