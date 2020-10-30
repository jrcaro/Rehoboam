import glob
import os

if __name__ == "__main__":
    path = '/home/jrcaro/darknet/data/rehoboam_test'

    os.chdir(path)
    images = [f for f in glob.glob('*.jpg')]
    print(images)

    with open('/home/jrcaro/darknet/data/rehoboam.txt', 'w') as f:
        for name in images:
            f.write('data/rehoboam_test/'+name+'\n')