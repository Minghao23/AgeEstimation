## Introduction
This project create a Telegram Bot who could converse with you and predict your gender and age.
It was implemented by two parts, Server and Client.
Server is responsible for machine learning tasks. It can be deployed in multiple machines and handle tasks concurrently.
Client is responsible for receiving and sending message to multiple users, and interacting with Server.
Telegram bot was chosen as our user interface, which can simply send images.
The core machine learning project is based on a pretrained Age-Gender-Estimation project. Some of its files were modified in this project.
Details are in Github: https://github.com/yu4u/age-gender-estimation

## Dependencies
- Python3.5+
- Keras2.0+
- scipy, numpy, Pandas, tqdm, tables, h5py
- dlib (for demo)
- OpenCV3

Tested on:
- macOS High Sierra, Python 3.6.5, Keras 2.0.2, Tensorflow 1.0.0, Theano 1.0.3

Actually you must make sure that you have already installed all the dependencies in requirements.txt

## Start

To start Server:
```sh
cd AgeEstimationServer
sh start.sh
```

To start Client:
```sh
cd AgeEstimationClient
python3 src/conversation.py
```

When first time to start Server, a pretrained model will download from internet automatically. This will spend some time.
