# Rehoboam

The goal of this work is to create a dashboard through which you can observe selected camera traffic of Málaga in real-time and count the number of vehicles in each direction to alert before the congestion occurs.
To achieve this, the system has to be capable of differentiating all the directions a vehicle can take. Therefore, will be 2 division: by possible directions (head-on, backside, left side and right side) and by vehicle type (car, bus, motorcycle and truck).
Thanks to these 16 classes the system will be able to count the number of vehicles in a street or avenue. If the number exceeds a threshold and maintains it during a certain time is possible that traffic congestion happens.

# Acknowledgement
This work has been carried out with the [AlexeyAB Darknet Repository](https://github.com/AlexeyAB/darknet) and the [hunglc007 tensorflow-yolov4-tflite Repository](https://github.com/hunglc007/tensorflow-yolov4-tflite).

In the next links you can found the image dataset used in the training (balance and imbalance) and testing process:
- [Training](https://drive.google.com/file/d/1tae3bzr62cZmOlF1AdXBuVX_IX0EVhro/view?usp=sharing)
- [Testing](https://drive.google.com/file/d/1VBQKJ-YShWmlzcEuSeNviBh5RO9ONUNq/view?usp=sharing)

# Table of contents

- [Rehoboam](#rehoboam)
- [Acknowledgement](#acknowledgement)
- [Table of contents](#table-of-contents)
- [Requirements](#requirements)
- [Installation](#installation)

# Requirements
* Linux
* Docker
* Python >= 3.6
* [Mongo Atlas Free Acount](https://www.mongodb.com/es)

# Installation 
Not tested in Windows systems.

1. Clone the repository to a path of your choice.
    ```
    git clone https://github.com/jrcaro/Rehoboam.git
    cd Rehoboam
    ```
2. Create a virtual environment with pip and activate it.
    ```
    python3 -m venv venv
    source venv/bin/activate
    ```
3. Upgrade pip and install all the libraries.
   ```
   pip3 install --upgrade pip
   pip3 install -r requirements.txt
   ``` 
4. Download the TensorFlow [checkpoint](https://drive.google.com/file/d/1_yCGycnnHANMcZ6bW6iB9YVDdmxoXwDV/view?usp=sharing) with the training data. Unzip and copy the folder to ```Rehoboam/data/YOLO/```.
5. In order to connect the Mongo database with the project, is necessary to change a few lines of [utils.py](https://github.com/jrcaro/Rehoboam/blob/cf7810b9db2ae897bb19e6cadb6f21559aa57b64/utils.py#L40-L42). Remember to add your IP address into the MongoDB whitelist.
    - In L40 add your user, password and name of your database.
    - In L41 the name of your database.
    - In L42 the name of your collection.

7. (Optional) In ```dashboard.py``` [L39 and L40](https://github.com/jrcaro/Rehoboam/blob/aa0c21da85c4e8844d58fbfb9fef311bfbcb836d/dashboard.py#L39-L40) is possible to modify the threshold number of vehicles in the same direction and span for traffic congestion message.   
8. From the root of the project, run:
    ```
    sudo docker-compose up
    ```
9. In a browser, navigate to ```localhost:9000```, click on the "Cluster" tab and "Add Cluster". Name it and paste ```zookeeper:2181``` in the field "Cluster Zookeeper Hosts". Click in save.
10. In another terminal, initialize the Kafka consumer. Remember to activate the environment first.
    ```
    python3 kafka_consumer.py
    ```
11. In other terminals, initialize the Dashboard and the Kafka producer. Remember to activate the environment first.
    ```
    python3 dashboard.py
    ```
12. In a browser, navigate to ```localhost:8050``` and enjoy :)
13. (Optional) If you try to reload the Kafka server and it crashes, execute these commands and try it again.
   ```
   sudo docker-compose rm -f
   sudo docker-compose pull
   ``` 
