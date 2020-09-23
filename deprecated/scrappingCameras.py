import requests
import shutil
from datetime import datetime
import time

def scrapingCameras(camera_id, save_path):
    """Download the image from a url created by the camera ide and save it

    Args:
        camera_id (list): list of cameras ids to download
        save_path (string): path where save the image
    """

    #Create the urls list
    urls = ['http://ctrafico.movilidad.malaga.eu/cst_ctrafico/camara10{}.jpg'.format(i)
                for i in camera_id]

    for i in range(20):
        for url,cid in zip(urls, camera_id):
            response = requests.get(url, stream=True)
            if response.status_code == requests.codes.ok:
                timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
                with open(save_path + '/camara10{}-'.format(cid) + timestamp + '.jpg', 'wb') as f:
                    response.raw.decode_content = True
                    shutil.copyfileobj(response.raw, f)
                    time.sleep(5)
                    #return response.content


if __name__ == "__main__":
    #Cameras district 1 -> training
    id_list = [59,63,70,73]

    #path = '/home/jrcaro/rehoboam/images/training/'
    path = '/home/jrcaro/images_test/'

    scrapingCameras(camera_id=id_list, save_path=path)


