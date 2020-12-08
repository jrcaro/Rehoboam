import os
import glob
import numpy as np
import cv2
import albumentations as A
import random
from shutil import copy, move
from tqdm import tqdm
import collections
import plotly.graph_objects as go
import pandas as pd
from rotate import yoloRotatebbox
from txt2xml import convert

# convert from opencv format to yolo format
# H,W is the image height and width
def cvFormattoYolo(corner, H, W):
    bbox_W = corner[3] - corner[1]
    bbox_H = corner[4] - corner[2]

    center_bbox_x = (corner[1] + corner[3]) / 2
    center_bbox_y = (corner[2] + corner[4]) / 2

    return corner[0], round(center_bbox_x / W, 6),\
            round(center_bbox_y / H, 6),\
            round(bbox_W / W, 6),\
            round(bbox_H / H, 6)

def split_string(x):
    return x.split('.')[0]

def add_extension(f):
    return f+'.txt'    

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

def compare_file_class(c_list):
    temp = np.setdiff1d(c_list, [])
    c1 = 0
    c2 = 0

    for i in temp:
        if i != '0' and i != '1' and i != '3':
            c1 = c1 + c_list.count(i)
        else:
            c2 = c2 + c_list.count(i)

    if c2 == 0:
        c1 = c1 + 10
    return c1 - c2

def find_aug_candidates(class_dict):
    txt_list = [f for f in glob.glob('*.txt')]

    # open files to read the classes
    candidates = {}
    chart_list = []
    candidates_order = collections.OrderedDict()

    print('Reading the classes')
    for txt in tqdm(txt_list):
        class_list = []
        with open(txt) as txt_fh:
            for line in txt_fh:
                class_list.append(line[0:2].strip())
                chart_list.append(line[0:2].strip())

        candidates[txt.split('.')[0]] = compare_file_class(class_list)

    #count the number of examples by class
    temp = np.setdiff1d(chart_list, [])
    count_class = {class_dict[i]: chart_list.count(i) for i in temp}
    count_class = collections.OrderedDict(sorted(count_class.items()))

    candidates_order = {k: v for k, v in sorted(candidates.items(),
                                                 key=lambda item: item[1],
                                                 reverse=True)}
    return candidates_order, count_class

#/home/jrcaro/rehoboam/test/
#/mnt/c/Users/QH273CN/Desktop/rehoboam/test/
def generate_bar_chart(count_, tags,
                        save_path='/home/jrcaro/Pictures/Rehoboam/', 
                        chart_name='imbalanced_distribution.png'):
    #show bar chart
    names = [tags[c] for c in count_.keys()]
    values = list(count_.values())

    fig = go.Figure([go.Bar(x=names, y=values, text=values, textposition='auto')])
    fig.update_layout(
        xaxis_tickangle=-45,
        title='Distribución de clases para entrenamiento',
        font_family="Arial"
        )
    #fig.show()
    print('Saving figure')
    fig.write_image(save_path + chart_name, width=1400, height=1000)

def chart_from_folder(path_f, tags, chart_title, class_f,
                        save_path='/home/jrcaro/Desktop/', 
                        chart_name='inference_distribution.png',
                        ):
    os.chdir(path_f)
    txt_files = [f for f in glob.glob('*.txt')]

    class_list = []
    print('Reading the classes')
    for txt in tqdm(txt_files):
        with open(txt) as txt_fh:
            for line in txt_fh:
                class_list.append(line[0:2].strip())

    #print(class_list)

    #count the number of examples by class
    temp = np.setdiff1d(class_list, [])
    #print(temp)
    count_class = {tags[i]: class_list.count(i) for i in temp}
    
    #show bar chart
    names = [class_f[c] for c in sorted(count_class.keys(), key=lambda x: x)]
    values = [count_class[c] for c in sorted(count_class.keys(), key=lambda x: x)]

    fig = go.Figure([go.Bar(x=names, y=values, text=values, textposition='auto')])
    fig.update_layout(
        xaxis_tickangle=-45,
        title=chart_title,
        font_family="Arial"
        )
    #fig.show()
    print('Saving figure')
    fig.write_image(save_path + chart_name, width=1400, height=1000)

def transform_candidates(candidates, transforms):
    print('Transform the candidates')
    for filename in tqdm(candidates):
        image = cv2.imread(filename+'.jpg')
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        random.seed(42)
        for k,v in transforms.items():
            augmented_image = v(image=image)['image']
            cv2.imwrite(filename+'_{}'.format(k)+'.jpg', augmented_image)
            copy(filename+'.txt', filename+'_{}'.format(k)+'.txt')
    return

def find_class_delete(to_find, conditions, angle, break_val, stop_class):
    txt_list = [f for f in glob.glob('*.txt')]

    count_c = 0
    find = []
    print('Finding class')
    for txt in tqdm(txt_list):
        class_list = []
        with open(txt) as txt_fh:
            for line in txt_fh:
                class_list.append(line[0:2].strip())
        temp = np.setdiff1d(class_list, [])
        if len(np.setdiff1d(class_list, to_find)) < len(temp):
            find.append(txt)
    
    print('Rewriting txt')
    for txt in tqdm(find):
        if count_c >= break_val:
            break

        split_ = txt.split('.')
        new_name = split_[0] + '_copy'
        copy(split_[0] + '.jpg', new_name+'.jpg')
        
        with open(txt, "r") as n:
            lines = n.readlines()
        
        with open(new_name+'.txt', "w+") as f:
            for x in lines:
                if x[0:2].strip() == stop_class:
                    count_c = count_c + 1

                if x[0:2].strip() not in conditions:
                    f.write(x)

        # initiate the class
        im = yoloRotatebbox(new_name, '.jpg', angle)

        bbox = im.rotateYolobbox()
        image = im.rotate_image()

        # to write rotateed image to disk
        cv2.imwrite(new_name + '_rotated_'+ str(angle) + '.jpg', image)

        file_name = new_name + '_rotated_'+ str(angle) + '.txt'
        if os.path.exists(file_name):
            os.remove(file_name)

        # to write the new rotated bboxes to file
        for i in bbox:
            with open(file_name, 'a') as fout:
                fout.writelines(' '.join(map(str, cvFormattoYolo(i,\
                                    im.rotate_image().shape[0],\
                                    im.rotate_image().shape[1]))) + '\n')
        
        os.remove(new_name+'.txt')
        os.remove(new_name+'.jpg')

def balance_classes(classes_dict, classes_tags, limit, bar_name):
    cand_balance, waste = find_aug_candidates(classes_dict)

    c_class = [0] * 16
    candidates = []
    class_conditions = []
    print('Balancing classes')
    for t,txt in tqdm(enumerate(cand_balance.keys())):
        class_list = []
        with open(txt+'.txt') as txt_fh:
            for line in txt_fh:
                class_list.append(line[0:2].strip())
        temp = np.setdiff1d(class_list, [])
        if len(np.setdiff1d(temp, class_conditions)) == len(temp):
            for i in temp:
                c_class[int(i)] = c_class[int(i)] + class_list.count(i)
            candidates.append(txt.split('.')[0])
        if any(j >= limit for j in c_class):
            for k,cl in enumerate(c_class):
                if cl >= limit and str(k) not in class_conditions:
                    class_conditions.append(str(k))
        if all(j >= limit for j in c_class):
            break
        #print(t, c_class, class_conditions)

    count_class_chart = {classes_dict[str(k)]: c for k,c in enumerate(c_class)}
    #generate_bar_chart(count_class_chart, classes_tags, chart_name=bar_name)

    return candidates

def create_copy(folder, files_test, files_train, tf):
    os.mkdir(folder)
    os.mkdir(folder+'/valid')
    os.mkdir(folder+'/train')

    print('Copying {} dataset'.format(folder))
    for fts in tqdm(files_test):
        if tf:
            move(fts+'.jpg', folder+'/valid/'+fts+'.jpg')
            move(fts+'.xml', folder+'/valid/'+fts+'.xml')
        else:
            copy(fts+'.jpg', folder+'/valid/'+fts+'.jpg')
            move(fts+'.txt', folder+'/valid/'+fts+'.txt')

    for ftr in tqdm(files_train):
        if tf:
            move(ftr+'.jpg', folder+'/train/'+ftr+'.jpg')
            move(ftr+'.xml', folder+'/train/'+ftr+'.xml')
        else:
            copy(ftr+'.jpg', folder+'/train/'+ftr+'.jpg')
            move(ftr+'.txt', folder+'/train/'+ftr+'.txt')

def split_train_valid(limit, val_ratio):
    print('Creating the xml files')
    os.chdir('Balanced')
    convert() #Cambiar ruta otro PC

    n_limit = limit*val_ratio

    txt_list = [f.split('.')[0] for f in glob.glob('*.txt')]
    
    random.seed(258)
    random.shuffle(txt_list)

    c_class = [0] * 16
    to_valid = []
    class_conditions = []
    print('Reading the balanced classes')
    for t, txt in tqdm(enumerate(txt_list)):
        class_list = []
        with open(txt+'.txt') as txt_fh:
            for line in txt_fh:
                class_list.append(line[0:2].strip())
        temp = np.setdiff1d(class_list, [])
        if len(np.setdiff1d(temp, class_conditions)) == len(temp):
            for i in temp:
                c_class[int(i)] = c_class[int(i)] + class_list.count(i)
            to_valid.append(txt)
        if any(j >= n_limit for j in c_class):
            for k,cl in enumerate(c_class):
                if cl >= n_limit and str(k) not in class_conditions:
                    class_conditions.append(str(k))
        if all(j >= n_limit for j in c_class):
            break
        #print(t, c_class, class_conditions)
    
    to_train = list(np.setdiff1d(txt_list,to_valid))
    for fol in ['YOLO', 'TF']:
        create_copy(fol, to_valid, to_train, fol=='TF')


def main(lim, th, brk, brk_cl, chart_name):
    #set the images folder
    path = '/home/jrcaro/images'
    #path = '/mnt/c/Users/QH273CN/Downloads/obj'
    #path for data
    path_data = '../data/rehoboam_data.xlsx'

    # Classes in the images
    tag_df = pd.read_excel(path_data, sheet_name='classes')
    classes_tags = dict(zip(tag_df.tag, tag_df.name))
    tag_df.index = tag_df.index.map(str)
    classes_ = dict(zip(tag_df.index, tag_df.tag))

    copy_class = ['6', '7', '14']

    transforms = {
        'blur_0': A.Blur(blur_limit=2, always_apply=True, p=1),
        'blur_1': A.Blur(blur_limit=4, always_apply=True, p=1),
        'blur_2': A.Blur(blur_limit=6, always_apply=True, p=1),
        'blur_3': A.Blur(blur_limit=8, always_apply=True, p=1),
        'blur_gauss_0': A.GaussianBlur(blur_limit=1, p=1),
        'blur_gauss_1': A.GaussianBlur(blur_limit=(1,3), p=1),
        'blur_gauss_2': A.GaussianBlur(blur_limit=(3,5), p=1),
        'blur_gauss_3': A.GaussianBlur(blur_limit=(5,7), p=1),
        'blur_gauss_4': A.GaussianBlur(blur_limit=(7,9), p=1),
        'blur_glass_2': A.GlassBlur (sigma=0.1, max_delta=3, always_apply=True, p=1),
        'blur_glass_3': A.GlassBlur (sigma=0.9, max_delta=2, always_apply=True, p=1),
        'blur_glass_4': A.GlassBlur (sigma=1.7, max_delta=1, always_apply=True, p=1),
        'blur_motion_0': A.MotionBlur(blur_limit=3, p=1),
        'blur_motion_1': A.MotionBlur(blur_limit=(3,5), p=1),
        'blur_motion_2': A.MotionBlur(blur_limit=(5,7), p=1),
        'blur_motion_3': A.MotionBlur(blur_limit=(7,9), p=1),
        'blur_motion_4': A.MotionBlur(blur_limit=(9,11), p=1),
        'bright_0': A.RandomBrightness(limit=(-0.2,-0.1), p=1),
        'bright_1': A.RandomBrightness(limit=(-0.3,-0.2), p=1),
        'bright_2': A.RandomBrightness(limit=(-0.4,-0.3), p=1),
        'bright_5': A.RandomBrightness(limit=(0.1,0.2), p=1),
        'bright_6': A.RandomBrightness(limit=(0.2,0.3), p=1),
        'bright_7': A.RandomBrightness(limit=(0.3,0.4), p=1),
        'ch_drop_0': A.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=1),
        'ch_drop_1': A.ChannelDropout(channel_drop_range=(1, 1), fill_value=128, p=1),
        'ch_drop_2': A.ChannelDropout(channel_drop_range=(1, 1), fill_value=255, p=1),
        'ch_drop_3': A.ChannelDropout(channel_drop_range=(2, 2), fill_value=0, p=1),
        'ch_drop_4': A.ChannelDropout(channel_drop_range=(2, 2), fill_value=128, p=1),
        'ch_sh': A.ChannelShuffle(p=1),
        'clahe_0': A.CLAHE(clip_limit=(1,2.0), tile_grid_size=(8, 8), p=1),
        'clahe_1': A.CLAHE(clip_limit=(2,3.6), tile_grid_size=(7, 7), p=1),
        'clahe_2': A.CLAHE(clip_limit=(3.6,9), tile_grid_size=(2, 2), p=1),
        'coarse_drop': A.CoarseDropout(p=1),
        'compress_0': A.JpegCompression(quality_lower=99, quality_upper=100, p=1),
        'compress_1': A.JpegCompression(quality_lower=59, quality_upper=60, p=1),
        'compress_2': A.JpegCompression(quality_lower=39, quality_upper=40, p=1),
        'compress_3': A.JpegCompression(quality_lower=19, quality_upper=20, p=1),
        'compress_4': A.JpegCompression(quality_lower=0, quality_upper=1, p=1),       
        'contr_0': A.RandomContrast(limit=(-0.2,-0.1), p=1),
        'contr_1': A.RandomContrast(limit=(-0.3,-0.2), p=1),
        'contr_2': A.RandomContrast(limit=(-0.4,-0.3), p=1),
        'contr_5': A.RandomContrast(limit=(0.1,0.2), p=1),
        'contr_6': A.RandomContrast(limit=(0.2,0.3), p=1),
        'contr_7': A.RandomContrast(limit=(0.3,0.4), p=1),
        'cut_0': A.Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, p=1),
        'cut_1': A.Cutout(num_holes=10, max_h_size=20, max_w_size=20, fill_value=0, p=1),
        'fog': A.RandomFog(fog_coef_upper=0.5, alpha_coef=0.1,p=1),
        'hue_0': A.HueSaturationValue(hue_shift_limit=(50,100), sat_shift_limit=50, val_shift_limit=20, p=1),
        'hue_1': A.HueSaturationValue(hue_shift_limit=(100,150), sat_shift_limit=50, val_shift_limit=20, p=1),
        'hue_2': A.HueSaturationValue(hue_shift_limit=(150,200), sat_shift_limit=50, val_shift_limit=20, p=1),
        'hue_3': A.HueSaturationValue(hue_shift_limit=(200,250), sat_shift_limit=50, val_shift_limit=20, p=1),
        'hue_4': A.HueSaturationValue(hue_shift_limit=(250,300), sat_shift_limit=50, val_shift_limit=20, p=1),
        'gray': A.ToGray(p=1),
        'invert': A.InvertImg(p=1),
        'noise_0': A.MultiplicativeNoise(multiplier=0.5, p=1),
        'noise_1': A.MultiplicativeNoise(multiplier=[0.5, 1.5], per_channel=True, p=1),
        'noise_2': A.MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, p=1),
        'noise_3': A.MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, per_channel=True, p=1),
        'sepia': A.ToSepia(p=1),
        'snow_0': A.RandomSnow(brightness_coeff=1.5,p=1),
        'snow_1': A.RandomSnow(brightness_coeff=2,p=1),
        'sun_0': A.RandomSunFlare(src_radius=100,p=1),
        'sun_1': A.RandomSunFlare(src_radius=200,p=1),
        'sun_2': A.RandomSunFlare(src_radius=300,p=1),
    }

    os.chdir('/home/jrcaro/TFM/Imagenes/images_test')
    '''if os.path.exists('classes.txt'):
        os.remove('classes.txt')

    delete_test()    

    find_class_delete(copy_class,
                np.setdiff1d(list(classes_.keys()), copy_class+['12']), 15, brk, brk_cl)
    
    cand, count = find_aug_candidates(classes_)
    cand_list = [k for k, v in cand.items() if v >= th]
    #generate_bar_chart(count, tags)

    transform_candidates(cand_list, transforms)
    final_files = balance_classes(classes_, tags, lim, chart_name)

    os.mkdir('Balanced')

    for f in final_files:
        copy(f+'.txt', 'Balanced/{}'.format(f+'.txt'))
        copy(f+'.jpg', 'Balanced/{}'.format(f+'.jpg'))

    split_train_valid(lim, 0.2)'''
    #delete_test() 
    chart_from_folder(path_f='/home/jrcaro/TFM/Imagenes/images_test', tags=classes_, 
                    chart_title='Distribucion para inferencia', class_f=classes_tags)

if __name__ == "__main__":
    limit = 3000
    break_class = '14'
    break_val = 60
    threshold = 0

    main(limit,
        threshold,
        break_val,
        break_class,
        'inference_distribution_2.png')
    
