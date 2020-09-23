from kafka import KafkaConsumer
from PIL import Image
import cv2
import avro.schema
import io
from avro.io import DatumReader, BinaryDecoder

# Decode messages
def decode(msg_value, reader):
    message_bytes = io.BytesIO(msg_value)
    decoder = BinaryDecoder(message_bytes)
    event_dict = reader.read(decoder)
    return event_dict

def main():
    schema = avro.schema.Parse(open("test/scheme.avsc", "r").read())
    reader = DatumReader(schema)
    consumer = KafkaConsumer('avro', bootstrap_servers=['localhost:9094'],\
         value_deserializer=lambda m: decode(m, reader))    

    for msg in consumer:
        print(msg.value)
        '''img = io.BytesIO(msg.value)
        img_pil = Image.open(img).save('cosa.jpg')'''

if __name__ == "__main__":
    main()