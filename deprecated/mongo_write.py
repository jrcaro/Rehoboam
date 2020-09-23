from datetime import datetime
from random import randint
import pymongo
from time import sleep

def mongoConnect():
    """Manage the Mongo connection

    Returns:
        (pymongo collection): PyMongo collection to write in
    """

    client = pymongo.MongoClient(
        "mongodb+srv://admin:admin@rehoboam.kdafu.gcp.mongodb.net/Rehoboam?retryWrites=true&w=majority")
    db = client.rehoboam_results
    collection = db.classes_result

    return collection

def mongo_write(classes_, district_id=1, camera_id=17):
    mongo_col = mongoConnect()

    for _ in range(100):
        result_dict = {class_: randint(0,10) for class_ in classes_}

        result = {
            'district_id': district_id,
            'camera_id': camera_id,
            'timestamp': datetime.utcnow(),
            'results': result_dict
            }

        mongo_col.insert_one(result).inserted_id
        sleep(5)

if __name__ == "__main__":
    path_class = '/home/jrcaro/labelImg/data/predefined_classes.txt'

    with open(path_class) as f:
        class_dict = {i: line.split('\n')[0] for i,line in enumerate(f)}
    class_list = list(class_dict.values())
    mongo_write(classes_=class_list)