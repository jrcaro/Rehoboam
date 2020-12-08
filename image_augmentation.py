import os
import albumentations as A
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White

def yolo2album(bbox_yolo, h, w):

    x_center, y_center, w_bb, h_bb = bbox_yolo
    y_min = (y_center*2*h - h_bb*h)/2
    y_max = h_bb*h + y_min
    x_min = (x_center*2*w - w_bb*w)/2
    x_max = w_bb*w + x_min

    return [x_min, y_min, x_max, y_max]

def yolo2COCO(bbox_yolo, h, w):

    x_center, y_center, w_nbb, h_nbb = bbox_yolo
    w_bb = round(w_nbb * w)
    h_bb = round(h_nbb * h)
    x_min = round((x_center - (w_nbb / 2)) * w)
    y_min = round((y_center - (h_nbb / 2)) * h)

    return [x_min, y_min, w_bb, h_bb]

def COCO2yolo(bbox_coco, h, w):

    x_min, y_min, w_bb, h_bb = bbox_coco
    w_nbb = w_bb / w
    h_nbb = h_bb / h
    x_center = w_nbb/2 + x_min/w
    y_center = h_nbb/2 + y_min/h

    return [round(x_center, 6), round(y_center, 6), round(w_nbb, 6), round(h_nbb, 6)]

def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)

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

def count_classes(list_txt):
    count = [0] * 16

    for f in list_txt:
        with open(f) as fh:
            lines = fh.readlines()
            for l in lines:
                count[int(l.split(' ')[0])] += 1

    return count

def find_candidates():

    list_txt = [x for x in glob.glob('*.txt')]
    candidates = []

    for f in list_txt:
        with open(f) as fh:
            lines = fh.readlines()
            for l in lines:
                if int(l.split(' ')[0]) in (7, 11, 15):
                    candidates.append(f)

    return candidates

def split_string(x):
    return x.split('.')[0]

def delete_test():
    if 'Balanced' in os.listdir():
        shutil.rmtree('Balanced')
    txt_files = [f for f in glob.glob('*.txt')]
    jpg_list = [f for f in glob.glob('*.jpg')]

    jpg_delete = np.setdiff1d(list(map(split_string, jpg_list)),
                    list(map(split_string, txt_files)))

    #remove the images with no text file associated
    print('Deleting images with no text file')
    for f in jpg_delete:
        if os.path.exists(f+'.jpg'):
            os.remove(f+'.jpg')

    print('Deleting augmented images')
    for f in txt_files:
        if len(f.split('_')) > 2:
            os.remove(f)
            os.remove(f.split('.')[0]+'.jpg')
            #os.remove(f.split('.')[0]+'.xml')

def txt2bbox(filename):
    bboxes = []
    category_ids = []

    with open(filename+'.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            category_ids.append(int(line.split(' ')[0]))
            bbox = [float(i) for i in line.split(' ')[1:]]
            bboxes.append(bbox)
            
    return bboxes, category_ids

def saveTxt(filename, bboxes, classes):
    for bbox, class_ in zip(bboxes, classes):
        if int(class_) == 3:
            class_ = 2
        elif int(class_) == 7:
            class_ = 6
        elif int(class_) == 11:
            class_ = 10
        elif int(class_) == 15:
            class_ = 14

        bbox_str = COCO2yolo(bbox, 288, 384)

        with open(filename+'_flip.txt', 'a') as f:
            f.write('{} {} {} {} {}\n'.format(
                class_, 
                str(bbox_str[0]),
                str(bbox_str[1]),
                str(bbox_str[2]),
                str(bbox_str[3])))


def flip_images():
    # Classes in the images
    tag_df = pd.read_excel('data/rehoboam_data.xlsx', sheet_name='classes')
    tags = dict(enumerate(tag_df.tag))

    os.chdir('/home/jrcaro/TFM/Imagenes/images_test')

    delete_test()

    candidates = find_candidates()

    transform = A.Compose([A.HorizontalFlip(1)], \
        bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

    for candidate in candidates:
        filename = candidate.split('.')[0]
        image = cv2.imread(filename + '.jpg')

        bboxes_coco = []
        bboxes_yolo, category_ids = txt2bbox(filename)
        print(candidate, bboxes_yolo)

        for bbox_yolo in bboxes_yolo:
            bboxes_coco.append(yolo2COCO(bbox_yolo, 288, 384))

        # Print bbox -> coco (x_min, y_min, x_max, y_max)
        #visualize(image, bboxes_coco, category_ids, tags)

        transformed = transform(image=image, bboxes=bboxes_coco, category_ids=category_ids)
        
        '''visualize(
            transformed['image'],
            transformed['bboxes'],
            transformed['category_ids'],
            tags,
        )'''    

        cv2.imwrite(filename+'_flip'+'.jpg', transformed['image'])
        saveTxt(filename, transformed['bboxes'], transformed['category_ids'])

def score_filename():
    os.chdir('/home/jrcaro/TFM/Imagenes/images_test')
    filenames = [f for f in glob.glob('*.txt')]

    class_count = count_classes(filenames)
    class_score = {i: 1/c for i,c in enumerate(class_count)}

    print(class_count)
    print(class_score)

    scores = {}

    for filename in filenames:
        sum_file = 0
        _,classes = txt2bbox(filename.split('.')[0])

        for c in classes:
           sum_file += class_score[c]

        scores[filename] = sum_file

    scores_sort = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}
    
    return scores_sort
        

if __name__ == "__main__":
    #flip_images()
    files_with_score = score_filename()

    print(files_with_score)

        
    
