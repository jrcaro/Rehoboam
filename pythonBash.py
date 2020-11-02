import subprocess

pathDarknet = '/home/jrcaro/TFM/darknet/darknet '
pathData = ' /home/jrcaro/TFM/darknet/data/rehoboam.data '
pathCfg = ' /home/jrcaro/TFM/darknet/yolov4-obj.cfg '
pathWeight = ' /home/jrcaro/TFM/darknet/data/yolov4-obj_last.weights '
pathImage = ' /home/jrcaro/TFM/darknet/data/rehoboam_test/camara103-07092020_111752.jpg '
bashCommand = pathDarknet + 'detector test' + pathData + pathCfg \
            + pathWeight + pathImage + '-i 0 -thresh 0.25'

process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()