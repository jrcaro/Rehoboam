# Rehoboam

Answer to traffic congestion applying object detection algorithm

# Table of contents

- [Rehoboam](#rehoboam)
- [Table of contents](#table-of-contents)
- [Requirements](#requirements)
- [Introduction](#introduction)
- [Installation](#installation)

# Requirements
* Linux
* Docker
* Python >= 3.6
* [Mongo Atlas Acount](https://www.mongodb.com/es)

# Introduction
The goal of this work is to create a dashboard through you can observed a selected camera traffic of Málaga in real time and count the number of vehicles in each direction in order to alert before the congestion occur.
To achieve this, the system has to be capable of differentiate all the directions a vehicle can take. Therefore, will be 2 division: by possible directions (head-on, backside, left side and right side) and by vehicle type (car, bus, motorcycle and truck).
Thanks to this 16 classes the system will be able to count he number of vehicles in a street or avenue. If the number exceed a threshold and maintain it during a certain time is possible that a traffic congestion happens.

# Installation 
Not tested in Windows systems.

1. Clone the repository to a path of your choice.
    ```
    git clone https://github.com/jrcaro/Rehoboam.git
    cd Rehoboam
    ```
2. Create a virtual enviroment with pip and activate it.
    ```
    python3 -m venv venv
    source venv/bin/activate
    ```
3. Upgrade pip and install all the libraries.
   ```
   pip3 install --upgrade pip
   pip3 install -r requirements.txt
   ``` 
4.  In order to connect the Mongo database with the project, is neccesary to change a few lines of [utils.py](https://github.com/jrcaro/Rehoboam/blob/cf7810b9db2ae897bb19e6cadb6f21559aa57b64/utils.py#L40-L42). Remember to add your IP address into the MongoDB whitelist.
    - In L40 add your user, password and name of your database.
    - In L41 the name of your database.
    - In L42 the name of your collection.
5.  From the root of the project, run:
    ```
    sudo docker-compose up
    ```
6.  In a browser, navigate to ```localhost:9000```, click on the "Cluster" tab and "Add Cluster". Name it and paste ```zookeeper:2181``` in the field "Cluster Zookeeper Hosts". Click in save.
7.  In other terminal, initialize the Kafka consumer. Remember activate the enviroment first.
    ```
    python3 kafka_consumer.py
    ```
8.  In other terminal, initialize the Dashboard and the Kafka producer. Remember activate the enviroment first.
    ```
    python3 dashboard.py
    ```
9.  In a browser, navigate to ```localhost:8050``` and enjoy :)
10. (Optional) If you try to reload the Kafka server and it crashs, execute this commnads an try it again.
   ```
   sudo docker-compose rm -f
   sudo docker-compose pull
   ``` 
