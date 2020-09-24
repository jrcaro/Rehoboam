from lxml import etree
from PIL import Image
import csv
import os
from tqdm import tqdm
import glob

# Change this labels
labels = [
    'coche_frontal',
    'coche_trasero',
    'coche_lateral_d',
    'coche_lateral_i',
    'camion_frontal',
    'camion_trasero',
    'camion_lateral_d',
    'camion_lateral_i',
    'moto_frontal',
    'moto_trasero',
    'moto_lateral_d',
    'moto_lateral_i',
    'autobus_frontal',
    'autobus_trasero',
    'autobus_lateral_d',
    'autobus_lateral_i'
]
global label
label = ''

def csvread(fn):
    with open(fn, 'r') as csvfile:
        list_arr = []
        reader = csv.reader(csvfile, delimiter=' ')

        for row in reader:
            list_arr.append(row)
    return list_arr


def convert_label(txt_file):
    global label
    for i in range(len(labels)):
        if txt_file[0] == str(i):
            label = labels[i]
            return label

    return label

def extract_coor(txt_file, img_width, img_height):
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

    return x_min_rect, x_max_rect, y_min_rect, y_max_rect

def convert(IMG_PATH='.',
            save_path='.',
            txt_folder = '.'):
    
    fw = [f for f in glob.glob('*.txt')]

    for line in tqdm(fw):
        root = etree.Element("annotation")

        # try debug to check your path
        img_style = IMG_PATH.split('/')[-1]
        img_name = line.split('.')[0]
        image_info = IMG_PATH + "/" + img_name + '.jpg'
        img_txt_root = txt_folder + "/" + img_name
        # print(img_txt_root)

        # print(txt_path)
        txt_file = csvread(line)
        ######################################

        # read the image  information
        img_size = Image.open(image_info).size

        img_width = img_size[0]
        img_height = img_size[1]
        img_depth = Image.open(image_info).layers
        ######################################

        folder = etree.Element("folder")
        folder.text = "%s" % (img_style)

        filename = etree.Element("filename")
        filename.text = "%s" % (img_name)

        path = etree.Element("path")
        path.text = "%s" % (IMG_PATH)

        source = etree.Element("source")
        ##################source - element##################
        source_database = etree.SubElement(source, "database")
        source_database.text = "Unknown"
        ####################################################

        size = etree.Element("size")
        ####################size - element##################
        image_width = etree.SubElement(size, "width")
        image_width.text = "%d" % (img_width)

        image_height = etree.SubElement(size, "height")
        image_height.text = "%d" % (img_height)

        image_depth = etree.SubElement(size, "depth")
        image_depth.text = "%d" % (img_depth)
        ####################################################

        segmented = etree.Element("segmented")
        segmented.text = "0"

        root.append(folder)
        root.append(filename)
        root.append(path)
        root.append(source)
        root.append(size)
        root.append(segmented)

        for ii in range(len(txt_file)):
            label = convert_label(txt_file[ii][0])
            x_min_rect, x_max_rect, y_min_rect, y_max_rect = extract_coor(
                txt_file[ii], img_width, img_height)

            object = etree.Element("object")
            ####################object - element##################
            name = etree.SubElement(object, "name")
            name.text = "%s" % (label)

            pose = etree.SubElement(object, "pose")
            pose.text = "Unspecified"

            truncated = etree.SubElement(object, "truncated")
            truncated.text = "0"

            difficult = etree.SubElement(object, "difficult")
            difficult.text = "0"

            bndbox = etree.SubElement(object, "bndbox")
            #####sub_sub########
            xmin = etree.SubElement(bndbox, "xmin")
            xmin.text = "%d" % (x_min_rect)
            ymin = etree.SubElement(bndbox, "ymin")
            ymin.text = "%d" % (y_min_rect)
            xmax = etree.SubElement(bndbox, "xmax")
            xmax.text = "%d" % (x_max_rect)
            ymax = etree.SubElement(bndbox, "ymax")
            ymax.text = "%d" % (y_max_rect)
            #####sub_sub########

            root.append(object)
            ####################################################

        file_output = etree.tostring(root, pretty_print=True, encoding='UTF-8')
        # print(file_output.decode('utf-8'))
        ff = open(save_path+'/%s.xml' % (img_name), 'w', encoding="utf-8")
        ff.write(file_output.decode('utf-8'))

if __name__ == "__main__":
    os.chdir('test')
    convert()
