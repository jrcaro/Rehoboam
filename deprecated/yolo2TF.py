from utils import save_tf, scrapingCameras, detect
import pandas as pd
import os
import glob
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import numpy as np
from PIL import Image

def extract_coor(line, img_width, img_height):

    txt_file = line.split(' ')
    x_rect_mid = float(txt_file[1])
    y_rect_mid = float(txt_file[2])
    width_rect = float(txt_file[3])
    height_rect = float(txt_file[4])

    x_min_rect = ((2 * x_rect_mid * img_width) - (width_rect * img_width)) / 2
    x_max_rect = ((2 * x_rect_mid * img_width) + (width_rect * img_width)) / 2
    y_min_rect = ((2 * y_rect_mid * img_height) -
                  (height_rect * img_height)) / 2
    y_max_rect = ((2 * y_rect_mid * img_height) +
                  (height_rect * img_height)) / 2

    coord[txt_file[0]] = [x_min_rect, x_max_rect, y_min_rect, y_max_rect]

    return coord

if __name__ == "__main__":
    with open('data/models/YOLO/obj.names') as f:
        names_dict = {i: line.split('\n')[0] for i,line in enumerate(f)}

    config = {
            'weights': './data/models/YOLO/yolov4_imbalanced.weights',
            'input_size': 416,
            'score_thres': 0.5,
            'model': 'yolov4',
            'weights_tf': './checkpoints/yolov4_imbalanced',
            'output_path': 'result.jpg',
            'iou': 0.45
        }

    #save_tf(config)

    '''data_df = pd.read_excel('data/rehoboam_data.xlsx', sheet_name='cameras')
    #data_df = data_df.set_index('id_district', drop=True)
    data_df = data_df[data_df['readable']==1]
    data_df = data_df[data_df['id_district'] > 1]
    cameras_list = list(data_df['id_camera'])'''

    '''for _ in range(10):
        scrapingCameras(camera_id=cameras_list, save_path='/home/jrcaro/images_test', save=True)
    '''

    saved_model_loaded = tf.saved_model.load(
        config['weights_tf'], tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    files_folder = [i.split('.')[0] for i in glob.glob('/home/jrcaro/images_test/*.txt')]

    for img in files_folder[0:2]:
        coord = {}
        img_pil = Image.open(img+'.jpg')
        img_width = img_pil.size[0]
        img_height = img_pil.size[1]
        img_cv2 = np.array(img_pil)

        with open(img+'.txt') as txt_fh:
            for line in txt_fh:
                print(line)
                txt_box = extract_coor(line, img_width, img_height)
        print(txt_box)
        print(img)

        scores, classes, box = detect(parameters=config, infer_model=infer, image=img_cv2)
        print(box[0])
        real_classes = [i for i,j in zip(classes[0], scores[0]) if j > 0.0]
        
        unique, count = np.unique(real_classes, return_counts=True)
        result_dict = {names_dict[i]: int(j) for i,j in zip(unique, count)}
        print(result_dict)