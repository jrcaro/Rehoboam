# -*- coding: utf-8 -*-

from xml.dom import minidom
import os
import glob

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

        print ("wrote %s" % fname_out)



def main():
    convert_xml2yolo(lut)


if __name__ == '__main__':
    os.chdir('/home/jrcaro/images_test/')
    for f in glob.glob('*.xml'):
        os.remove(f)
    #main()