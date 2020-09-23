import requests
import shutil
from datetime import datetime
from kafka import KafkaProducer
from kafka import errors as Errors
import avro.schema
import io
from avro.io import DatumWriter, BinaryEncoder
from PIL import Image

def scrapingCameras(camera_id, save_path='./', save=False):
    """Download the image from a url created by the camera ide and save it

    Args:
        camera_id (list): list of cameras ids to download
        save_path (str, optional): path where save the image. Defaults './'
        save (boolean, optional): flag to save the image. Defaults False
    Returns:
      (bytes): image to process in bytes
    """

    # Create the urls list
    urls = ['http://ctrafico.movilidad.malaga.eu/cst_ctrafico/camara10{}.jpg'.
            format(i if i >= 10 else '0'+str(i)) for i in camera_id]

    for url, cid in zip(urls, camera_id):
        response = requests.get(url, stream=True)
        if response.status_code == requests.codes.ok:
            response.raw.decode_content = True
            if save:
                timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
                with open(save_path + '/camara10{}-'.format(cid) + timestamp + '.jpg', 'wb') as f:
                    shutil.copyfileobj(response.raw, f)
            return response.content

# Function to encode bytes
def encode(msg, writer):
    bytes_writer = io.BytesIO()
    encoder = BinaryEncoder(bytes_writer)
    writer.write(msg, encoder)
    raw_bytes = bytes_writer.getvalue()
    return raw_bytes

def main():
    data = scrapingCameras([17])
    schema = avro.schema.Parse(open("test/scheme.avsc", "r").read())
    writer = DatumWriter(schema)

    resource = {"image": data, "camera": str(17)}
    '''resources = []
    resources.append(resource)
    server = {'ip': "127.0.0.1", "port": 5683, "resources": resources}'''

    producer = KafkaProducer(bootstrap_servers=['localhost:9094'],\
                value_serializer=lambda m: encode(m, writer))    

    future = producer.send('avro', resource)

    #future=producer.send('masterbigdata',key=b"temp", value=b'temperature-value')

    try:
        future.get(timeout=60)  # Block until the message is sent
    except Errors.KafkaTimeoutError:
        print("Message could not be sent!")

    producer.flush()  # Wait until all pending messages are at least put on the network
    producer.close()  # Close the connection

if __name__ == "__main__":
    main()