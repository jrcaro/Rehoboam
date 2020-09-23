import os
import glob
from xml.dom import minidom
import numpy as np

def split_string(x):
    return x.split('.')[0]

def delete_files(all_lst, jpg_lst, xml_lst):
    temp = np.setdiff1d(all_lst, jpg_lst)
    no_name_files = np.setdiff1d(temp, xml_lst)

    to_delete = list(map(split_string, xml_lst))
    for d in to_delete:
        if os.path.exists(d):
            os.remove(d)

    temp2 = list(set(no_name_files)-set(to_delete))
    for f in temp2:
        os.rename(f, f+'.xml')

    xml_files = [f for f in glob.glob('*.xml')]

    jpg_delete = np.setdiff1d(list(map(split_string, jpg_lst)),
                    list(map(split_string, xml_files)))

    #remove the images with no text file associated
    for f in jpg_delete:
        if os.path.exists(f+'.jpg'):
            os.remove(f+'.jpg')

    return

def convert_coordinates(size, box):
    dw = 1.0/size[0]
    dh = 1.0/size[1]
    x = (box[0]+box[1])/2.0
    y = (box[2]+box[3])/2.0
    w = box[1]-box[0]
    h = box[3]-box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)


def convert_xml2yolo( lut ):

    for fname in glob.glob("*.xml"):
        
        xmldoc = minidom.parse(fname)
        
        fname_out = (fname[:-4]+'.txt')

        with open(fname_out, "w") as f:

            itemlist = xmldoc.getElementsByTagName('object')
            size = xmldoc.getElementsByTagName('size')[0]
            width = int((size.getElementsByTagName('width')[0]).firstChild.data)
            height = int((size.getElementsByTagName('height')[0]).firstChild.data)

            for item in itemlist:
                # get class label
                classid =  (item.getElementsByTagName('name')[0]).firstChild.data
                if classid in lut:
                    label_str = str(lut[classid])
                else:
                    label_str = "-1"
                    print ("warning: label '%s' not in look-up table" % classid)

                # get bbox coordinates
                xmin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmin')[0]).firstChild.data
                ymin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymin')[0]).firstChild.data
                xmax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmax')[0]).firstChild.data
                ymax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymax')[0]).firstChild.data
                b = (float(xmin), float(xmax), float(ymin), float(ymax))
                bb = convert_coordinates((width,height), b)
                #print(bb)

                f.write(label_str + " " + " ".join([("%.6f" % a) for a in bb]) + '\n')
        
        os.remove(fname)

        print ("wrote %s" % fname_out)

if __name__ == "__main__":
    path = '/mnt/c/Users/QH273CN/Downloads/data'
    os.chdir(path)

    lut={}
    lut["coche_frontal"] 	=0
    lut["coche_trasero"] 	=1
    lut["coche_lateral_d"]  =2
    lut["coche_lateral_i"]  =3
    lut["camion_frontal"]   =4
    lut["camion_trasero"] 	=5
    lut["camion_lateral_d"] =6
    lut["camion_lateral_i"] =7
    lut["moto_frontal"]     =8
    lut["moto_trasero"]     =9
    lut["moto_lateral_d"] 	=10
    lut["moto_lateral_i"]   =11
    lut["autobus_frontal"]  =12
    lut["autobus_trasero"]  =13
    lut["autobus_lateral_d"]=14
    lut["autobus_lateral_i"]=15

    all_files = os.listdir()
    jpg_files = [f for f in glob.glob('*.jpg')]
    xml_files = [f for f in glob.glob('*.xml')]

    delete_files(all_files, jpg_files, xml_files)
    convert_xml2yolo(lut)