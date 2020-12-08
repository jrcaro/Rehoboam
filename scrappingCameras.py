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
    urls = ['http://ctraficomovilidad.malaga.eu/cst_ctrafico/camara10{}.jpg'.\
            format(i if i >= 10 else '0'+str(i)) for i in camera_id]

    for url,cid in zip(urls, camera_id):
        for i in range(100):
            print('{} - {}'.format(cid, i))
            response = requests.get(url, stream=True)
            if response.status_code == requests.codes.ok:
                timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
                with open(save_path + '/camara10{}-'.format(cid if cid >= 10 else '0'+str(cid)) + timestamp + '.jpg', 'wb') as f:
                    response.raw.decode_content = True
                    shutil.copyfileobj(response.raw, f)
                    time.sleep(5)
                    #return response.content

if __name__ == "__main__":
    #Cameras district 1 -> training
    id_list = [73,74,13,15,31,32,48,52,72,54,55,56,57,63,21,59]

    #3,4,29,70,25,44,11,12,22,23,46,75,80,71,76,14,34,35,49,51,58,64,67,68,69,73,74,13,15,31,32,48,52,72,54,55,56,57,63,21,59
    #path = '/home/jrcaro/rehoboam/images/training/'
    path = '/home/jrcaro/TFM/Imagenes/images_test'

    scrapingCameras(camera_id=id_list, save_path=path)


