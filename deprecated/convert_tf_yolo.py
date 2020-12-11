import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import ConfigProto
from core.yolov4 import YOLO, decode, filter_boxes
import core.utils as utils
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import glob
import cv2
import tqdm
from utils import save_tf

BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White

warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def yolo_detect(parameters, image_path):
    #Load the tf model
    saved_model_loaded = tf.saved_model.load(
        parameters['weights_tf'], tags=[tag_constants.SERVING])
    infer_yolo = saved_model_loaded.signatures['serving_default']

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(tiny=False,
                                                            model=parameters['model'])
    input_size = parameters['input_size']
    img_pil = Image.open(image_path)
    img_cv2 = np.array(img_pil)
    image_data = cv2.resize(img_cv2, (input_size, input_size))
    image_data = image_data / 255.

    images_data = []
    for i in range(1):
        images_data.append(image_data)

    images_data = np.asarray(images_data).astype(np.float32)
    batch_data = tf.constant(images_data)
    pred_bbox = infer_yolo(batch_data)

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
    image = utils.draw_bbox(img_cv2, pred_bbox)
    image = Image.fromarray(image.astype(np.uint8))
    # image.show()
    #image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    #cv2.imwrite(parameters['output_path'], image)

    output = []
    prec = []
    for d in scores.numpy()[0]:
        if d > 0.0:
            prec.append(d)

    for i in range(len(prec)):
        temp = {}
        temp['class'] = int(classes.numpy()[0][i])
        temp['score'] = scores.numpy()[0][i]
        temp['box'] = boxes.numpy()[0][i].tolist()
        output.append(temp)
    
    return output

def tf_detect(model_path, image_path, score_thr,
            path_labels='data/models/label_map.pbtxt'):

    model = model_path.split('/')[2]
    PATH_TO_SAVED_MODEL = model_path + "/saved_model"
    # Load saved model and build the detection function
    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

    # Load label map data (for plotting)
    category_index = label_map_util\
        .create_category_index_from_labelmap(path_labels, use_display_name=True)

    print('Running inference for {}... '.format(image_path), end='')
    image_np = np.array(Image.open(image_path))
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]
    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)
    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections
    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    image_np_with_detections = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=100,
          min_score_thresh=score_thr,
          agnostic_mode=False)
    
    #res = Image.fromarray(image_np_with_detections)
    #res.save('data/res.jpg')

    ouput = []
    #print(detections['detection_boxes'].shape)
    for i in range(detections['detection_boxes'].shape[0]):
        temp = {}
        if detections['detection_scores'][i] > score_thr:
            temp['class'] = detections['detection_classes'][i]
            temp['score'] = detections['detection_scores'][i]
            temp['box'] = detections['detection_boxes'][i].tolist()
            ouput.append(temp)
        
    return ouput

def model2txt(filename, data, weight, height, names):
    with open(filename, 'w') as f:
        for d in data:
            y0 = int(d['box'][0]*height)
            x0 = int(d['box'][1]*weight) 
            y1 = int(d['box'][2]*height)
            x1 = int(d['box'][3]*weight)
            #print(x0, y0, x1, y1)
            #print(round(d['score'], 2))
            f.write(str(names[d['class']]) + " " + str(round(d['score'], 2)) + \
                " {} {} {} {}\n".format(x0, y0, x1, y1))

def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    print(bbox)
    y_min, x_min, y_max, x_max = bbox
    x_min, x_max, y_min, y_max = int(x_min*384), int(x_max*384), int(y_min*288), int(y_max*288)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=TEXT_COLOR, 
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    tf_models = ['data/models/SSD', 'data/models/faster_rcnn']

    config = {
        'weights': 'data/models/YOLO/yolov4_balanced_2.weights',
        'input_size': 416,
        'score_thres': 0.1,
        'model': 'yolov4',
        'weights_tf': 'data/models/YOLO/checkpoints/yolov4_balanced',
        'output_path': 'data/result.jpg',
        'iou': 0.45
    }

    #save_tf(config)

    path_names = 'data/models/YOLO/rehoboam.names'

    with open(path_names) as f:
        names_dict = {i: line.split('\n')[0] for i,line in enumerate(f)}

    count = 0
    for p in glob.glob('/home/jrcaro/TFM/Imagenes/images_test/*.jpg'):
        #print(p)
        width, height = Image.open(p).size
        name = p.split('/')[-1].split('.')[0]
        
        '''
        #print(p)
        yolo_data = yolo_detect(parameters=config, image_path=p)
        print(yolo_data)
        model2txt('/home/jrcaro/TFM/Extra/mAP/input/detection-results/{}.txt'.format(name),
                    yolo_data, width, height, names_dict)'''
        
        tf_data = tf_detect(tf_models[0], p, config['score_thres'])
        image = cv2.imread(p)
        '''visualize(image,
        [data['box'] for data in tf_data],
        [data['class'] for data in tf_data],
        names_dict)'''

        model2txt('/home/jrcaro/TFM/Extra/mAP/input/detection-results/{}.txt'.format(name),
                    tf_data, width, height, names_dict)

        print()
