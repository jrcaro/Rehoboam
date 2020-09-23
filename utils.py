import json
import time
import logging
import pymongo
import shutil
from datetime import datetime
import requests
import numpy as np
import cv2
import io
from PIL import Image
from kafka import KafkaProducer, KafkaConsumer
from kafka import errors as Errors
import avro.schema
from avro.io import DatumWriter, BinaryEncoder, BinaryDecoder
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import ConfigProto
from tensorflow.python.saved_model import tag_constants
import tensorflow as tf
from core.yolov4 import YOLO, decode, filter_boxes
import core.utils as utils
from core.config import cfg

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

'''
Mongo connection
'''

def mongoConnect():
    """Manage the Mongo connection

    Returns:
        (pymongo collection): PyMongo collection to write in
    """

    client = pymongo.MongoClient(
        "mongodb+srv://admin:admin@rehoboam.kdafu.gcp.mongodb.net/Rehoboam?retryWrites=true&w=majority")
    db = client.Rehoboam
    collection = db.result_collection

    return collection

'''
Scrapping camera function
'''

def scrapingCameras(camera_id, save_path='./', save=False):
    """Download the image from a url created by the camera ide and save it

    Args:
        camera_id ([int]): camera ids to download
        save_path ([str], optional): path where save the image. Defaults './'
        save ([boolean], optional): flag to save the image. Defaults False
    Returns:
      [bytes]: image to process
    """

    # Create the urls list
    url = 'http://ctrafico.movilidad.malaga.eu/cst_ctrafico/camara10{}.jpg'.\
            format(camera_id if camera_id >= 10 else '0'+str(camera_id))

    response = requests.get(url, stream=True)
    if response.status_code == requests.codes.ok:
        response.raw.decode_content = True
        if save:
            timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
            with open(save_path + '/camara10{}-'.format(camera_id) + timestamp + '.jpg', 'wb') as f:
                shutil.copyfileobj(response.raw, f)
        return response.content

'''
Tensorflow functions
'''

def save_tf(parameters):
    """Transform a darknet model of YOLO to a TensorFlow model

    Args:
        parameters (dictionary): input parameters
        - weights: path to the darknet weights
        - input_size: input size of the model
        - model: model to transform
        - weights_tf: path to save the tf weights
    Returns:
        [void]:
    """
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(
        tiny=False, model=parameters['model'])

    input_layer = tf.keras.layers.Input(
        [parameters['input_size'], parameters['input_size'], 3])
    feature_maps = YOLO(input_layer, NUM_CLASS, parameters['model'], False)
    bbox_tensors = []
    prob_tensors = []
    for i, fm in enumerate(feature_maps):
        if i == 0:
            output_tensors = decode(
                fm, parameters['input_size'] // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, 'tf')
        elif i == 1:
            output_tensors = decode(
                fm, parameters['input_size'] // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, 'tf')
        else:
            output_tensors = decode(
                fm, parameters['input_size'] // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, 'tf')
        bbox_tensors.append(output_tensors[0])
        prob_tensors.append(output_tensors[1])
    pred_bbox = tf.concat(bbox_tensors, axis=1)
    pred_prob = tf.concat(prob_tensors, axis=1)

    boxes, pred_conf = filter_boxes(pred_bbox, pred_prob, score_threshold=parameters['score_thres'], input_shape=tf.constant([
                                    parameters['input_size'], parameters['input_size']]))
    pred = tf.concat([boxes, pred_conf], axis=-1)
    model = tf.keras.Model(input_layer, pred)
    utils.load_weights(
        model, parameters['weights'], parameters['model'], False)
    model.summary()
    model.save(parameters['weights_tf'])


def detect(parameters, infer_model, image):
    """Detect the classes of a given image

     Args:
        parameters (dictionary): input parameters
        - input_size: input size of the model
        - score_thres: score threshold to draw a box
        - model: model to transform
        - weights_tf: path to save the tf weights
        - output_path: path where save the result image
        - iou: Intersection Over Union
        infer_model (tensorflow): loaded tensorflow model
        image (numpy array): image to detect
    Returns:
        tf.make_ndarray(proto_scores), tf.make_ndarray(proto_classes), pred_bbox
            [tfarray]: array with the precission of the class detection
            [tfarray]: array with the class predicted
    """
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(tiny=False,
                                                            model=parameters['model'])
    input_size = parameters['input_size']
    original_image = image
    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.

    images_data = []
    for i in range(1):
        images_data.append(image_data)

    images_data = np.asarray(images_data).astype(np.float32)
    batch_data = tf.constant(images_data)
    pred_bbox = infer_model(batch_data)

    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=parameters['iou'],
        score_threshold=parameters['score_thres']
    )

    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    image = utils.draw_bbox(original_image, pred_bbox)
    image = Image.fromarray(image.astype(np.uint8))
    # image.show()
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    cv2.imwrite(parameters['output_path'], image)

    proto_scores = tf.make_tensor_proto(scores)
    proto_classes = tf.make_tensor_proto(classes)
    
    return tf.make_ndarray(proto_scores), tf.make_ndarray(proto_classes)

'''
Kafka
'''

def encode(msg, writer):
    """Function to encode bytes for the Avro serilization

    Args:
        msg ([dict]): dictionary with the data for serialize
        writer ([DatumWriter]): special variable of Avro schema for the serilization

    Returns:
        [bytes]: message translate to bytes
    """

    bytes_writer = io.BytesIO()
    encoder = BinaryEncoder(bytes_writer)
    writer.write(msg, encoder)
    raw_bytes = bytes_writer.getvalue()

    return raw_bytes

def decode(msg_value, reader):
    """Function to decode the bytes from the Avro serilization

    Args:
        msg_value ([bytes]): message to deserialize
        reader ([DatumReader]): special variable of Avro schema for the deserilization

    Returns:
        [dict]: deserialize data dictionary
    """

    message_bytes = io.BytesIO(msg_value)
    decoder = BinaryDecoder(message_bytes)
    event_dict = reader.read(decoder)

    return event_dict

def kafkaProducer(camera_id, district_id,
                bootstrap_servers=['localhost:9094', 'localhost:9095'],
                input_topic="input_image",
                path_avro="data/scheme.avsc"):
    """Get the data from the cameras and send it throught a Kafka topic

    Args:
        camera_id ([int]): id number of the camera to download data
        district_id ([int]): id number of the camera district
        bootstrap_servers ([str], optional): Kafka brokers address. Defaults to ['localhost:9094', 'localhost:9095'].
        input_topic ([str]): name of the topic where queue the data. Defaults to "input_image"
        path_avro ([str]): path of the avro scheme
    Returns:
        [void]: 
    """

    #logging.basicConfig(level=logging.INFO)

    image = scrapingCameras(camera_id)

    schema = avro.schema.Parse(open(path_avro, "r").read())
    writer = DatumWriter(schema)

    resource = {"image": image, "camera": str(camera_id), "district": str(district_id)}

    producer = KafkaProducer(bootstrap_servers=['localhost:9094'], value_serializer=lambda m: encode(m, writer))    

    future = producer.send(input_topic, resource)
        
    try:
        future.get(timeout=15)  # Block until the message is sent
    except Errors.KafkaTimeoutError:
        print("Message could not be sent!")

    producer.flush()
    producer.close()

'''
Others
'''

def hour_dict(hour1, hour2):
    """Create a dictionary for the line chart figure

    Returns:
        [dict]: Dictionary with all the seconds of a day
    """
    data_chart = {}
    for i in range(hour1,hour2):
        for j in range(0,60):
            for k in range(0,60):
                date = datetime(2020,1,1,hour=i, minute=j, second=k)
                data_chart[date.time()] = None
    return data_chart