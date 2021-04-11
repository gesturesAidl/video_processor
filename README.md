# DEVICE CONTROL WITH GESTURES
Final project for the 2020-2021 Postgraduate course on Artificial Intelligence with Deep Learning, UPC School, authored by **Enrique González Terceño**, **Sofia Limon**, **Gerard Pons** and **Celia Santos**. 

Advised by **Amanda Duarte**.

## TABLE OF CONTENTS
[INTRODUCTION AND MOTIVATION](#introduction-and-motivation)

[DATASET](#dataset)

[ARCHITECTURE](#architecture)

[RESULTS](#results)

[END TO END IMPLEMENTATION](#end-to-end-implementation)

[HOW TO RUN THE PROGRAM](#how-to-run-the-program)


## INTRODUCTION AND MOTIVATION

How many times have you been watching TV and couldn't find the remote? Or you are cooking, eating... and your hands are too messy to interact with a device? Or maybe you would simply like to use devices more intuitively. We use gestures every day of our life, as our primary way of interacting with other humans. Our project emerges as a way to interact with devices in an easier, more convenient manner.

We have created a gesture recognition system that works by capturing videos with a camera, to perform basic tasks on a personal device. But our idea is that it can be further developed to different environments, like for virtual assistants or home appliances, or in a car (instead of taking your eyes off the road, you can control the navigation system a gesture and avoid any potential risks), or in a medical environment (for instance for a doctor to explore a radiography in a middle of a surgery).


## DATASET

The data we used to train and validate our model was from the [Jester Dataset](https://20bn.com/datasets/jester), which is a label collection of videos of humans performing hand gestures. The data is given in the JPG frames of the videos, which were recorded at 12 fps. Specifically, the dataset contains 150k videos of 27 different classes. As the goal of our project was to control basic functionalities of a computer, we decided to reduce the number of classes and to choose the gestures which made more sense from a control point of view. The used classes and the number of samples of each one are:

|          Gesture         | Samples |
|:------------------------:|:-------:|
|    Doing Other Things    |  12416  |
|        No Gesture        |   5344  |
| Sliding Two Fingers Down |   5410  | 
|  Sliding Two Fingers Up  |   5262  | 
|         Stop Sign        |   5413  | 
|       Swiping Left       |   5160  | 
|       Swiping Right      |   5066  | 
|         Thumb Up         |   5457  | 
|  Turning Hand Clockwise  |   3980  |

The first two classes, Doing Other Things and No Gesture, were added to our list of classes in order to have basic states when we are not trying to control the computer.

![alt text](https://github.com/gesturesAidl/video_processor/blob/main/images/jester-v1.gif?raw=true)

## ARCHITECTURE

To better understand the model’s architecture, the general pipeline should be first briefly explained. First of all, when an RGB video is received its optical flow is computed and features are extracted from both videos using an I3D Network. Then, these features are fed into a Neural Network, whose output are the probabilities for each class.

![alt text](https://github.com/gesturesAidl/video_processor/blob/main/images/architecture.png?raw=true)

### Optical Flow:
We computed the dense Optical Flow with the Farneback’s algorithm implementation of OpenCV. As the videos of the dataset have a low resolution and experience a lot of lightning changes, the Optical Flow results showed some imperfections, which we tried to solve by different approaches (MAYBE EXPLAIN THEM? :bangbang: ) . Unfortunately, we could not find how to correct them, and decided to move on with the Optical Flow we had in order to be able to continue advancing with the project.

![alt text](https://github.com/gesturesAidl/video_processor/blob/main/images/rgb_flow.gif?raw=true)

### I3D
In order to extract features from the videos (both RGB and Optical Flow) we used an Inflated 3D Network (I3D), as it is a widely used network for video classification, able to learn spatiotemporal information from the videos. In our case, the chosen I3D had weight initialization from a ResNet50 network trained on Imagenet, and was also pre-trained on the action recognition dataset Kinetics-400, since it’s an state-of-the-art model with a good balance between accuracy and computational requirements. The model is provided by [GluonCV](https://cv.gluon.ai/) and runs on MXnet.

### Neural Network
The RGB and the Optical Flow videos go through the previously mentioned I3D network, and features for both of them are obtained. One of the decisions we had to make is how to join them, and we explored various possibilities based on the ones mentioned in a [paper](http://vision.soic.indiana.edu/papers/extremelylow2018wacv.pdf) of action recognition for low resolution videos:
* Addition of the I3D features
* Concatenation of the I3D features
* Maximal value retention for each I3D features
* Addition of the logits from two different networks

The first three methods are based on the use of a single NN to classify the features, whereas the last one uses two different NNs, one for the RGB features and another one for the Optical Flow ones. The method with which the features are joined can have great impact on our network, as the number of layers, network parameters and hyperparameters to tune changes between them, so deciding which one to use was crucial to be able to start with an appropriate training of the network.

Hence, we tested the different methods with similar standard networks. We obtained the results shown on the figure below, and as we can see the performance is quite similar for all of the methods, with the concatenation (Concat) and logit addition (Sum after) methods performing slightly better than the other ones.

To decide which method to use we explored the number of parameters, to try to minimize computational time/cost, and of hyperparameters, to account for the tunability of the model. Regarding the parameters of the networks, both have a very similar number of them, with a difference on parameters of three orders of magnitude lower than the total number of parameters, so it was considered negligible. :heavy_exclamation_mark:With respect to the tunability, the method that adds up the logits has nearly the double of hyperparameters as the concatenation method, so we decided to use the former in our architecture.:heavy_exclamation_mark:

![alt text](https://github.com/gesturesAidl/video_processor/blob/main/images/feature_join.png?raw=true)

Hence, a neural network architecture with two streams was [created](https://github.com/gesturesAidl/video_processor/blob/main/app/GesturesAnalyzer/Classifier2stream.py) and trained, as will be commented in the [results section](#results). After the addition of the logits, the result goes through a LogSoftMax layer, and its output are the probabilities for each class.



## RESULTS

### FIRST APPROACH: RGB VIDEOS

The first way we explored to address the classification task was using only the extracted features from the RGB videos to classify them. After some training and hyperparameter tuning, the obtained accuracy was around 70%, which was still far from our desired accuracy values. To try to understand better where the model was struggling, we computed the confusion matrix of the predictions and some revealing results where found: the model encountered difficulties when differentiating the gestures that are the same movement but in different directions (i.e. Swiping left/Swiping right).

To address that problem, we thought that we could capture better the temporal and directional information by computing the Optical Flow of the videos and extract features from them.

![alt text](https://github.com/gesturesAidl/video_processor/blob/main/images/rgb.png?raw=true)

### SECOND APPROACH: OPTICAL FLOW VIDEOS

After training the model with only the Optical Flow features, following the same steps that had been done during the first approach, it was observed that while the accuracy of the whole model diminished, the confusion between the troublesome gestures was reduced, confirming the hypothesis that better directional information was captures although the whole classification was far worst. 

![alt text](https://github.com/gesturesAidl/video_processor/blob/main/images/flow.png?raw=true)

### THIRD APPROACH: TWO STREAM (RGB AND OPTICAL FLOW VIDEOS)

From the First and second approaches, we decided to use a model with two streams, one for RGB videos to help with the general classification task and the other one for the Optical Flow to address the confusion problem, and combine them as previously stated on the [architecture section](#architecture). Doing so doubles the amount of data and computer time/cost but was done in the hope of the model being able to keep the best parts of both approaches and yield better results.

#### Hyperparameter tuning

To try to get the best results possible, an extensive hyperparameter tuning was performed (note that it was also done in the first and second approach). In this case, the tuned parameters were: 

|          Parameter         | Value Range |
|:--------------------------:|:-----:|
|    Learning Rate           | loguniform(log(1e-4), log(1e-2)) |
|        Batch Size        |   choice(8, 16, 32, 64, 128, 256)  |
| RGB Hidden Layer |   choice(128, 256, 512, 1024, 2048) | 
|  Flow Hidden Layer  |  choice(128, 256, 512, 1024, 2048)  | 

:bangbang: WE SHOULD ADD A LITTLE EXPLANATION OF THE SCHEDUELRS OR SEARCH ALGORITHM USED, I DON'T REMEMBER IT WELL ENOUGH TO EXPLAIN THEM. ENRIQUE SEND HELP
:bangbang: PLOTS AND CONFUSSION NEED TO BE ADDED, WAITING FOR CONFIRMATION OF WHICH ONE (THE BEST)

As it can be observed on the figure, our hypothesis was true and the network managed to learn appropriately from the two streams of data and improved the overall accuracy, which went from 70% of the RGB videos and 57% of the Optical Flow ones to 83.1%, a significant increase.

### FURTHER EXPLORATION: DATA AUGMENTATION

We tried to further increase the accuracy by performing some data augmentation, keeping in mind our dataset and data treatment restrictions: 
* As we were working with Optical Flow, we could not apply Gaussian Blurs or other image manipulation techniques that involved creating artificial motion as the resulting Optical Flow turned out to be completely useless.
* As we had direction-dependant gestures, horizontal flips had to be discarded.
Hence we decided to perform data augmentation by applying a 15% zoom to the videos and also contrast, saturation and brightness changes.

Unfortunately, the results obtained did not show an improvement from the best models we had, so further exploration of the technique was discarded.

## END TO END IMPLEMENTATION

As stated in the introduction, our project goal was not only to train a working classifier but to use it to control a device. In our case, we decided to control our personal computer mapping some hand gestures to actions that will be displayed in a file window on a computer with a Linux OS GUI. 


|          Gesture         | Action  |
|:------------------------:|:-------:|
|         Thumb Up         |   Create a file  | 
|         Stop Sign        |  Close opened file  | 
|  Sliding Two Fingers Up  |   Create directory  | 
| Sliding Two Fingers Down |   Remove all in directory  | 
|       Swiping Left       |   Move to previous directory  | 
|       Swiping Right      |  Move to forward directory  | 
|  Turning Hand Clockwise  |   Close info/error popup  |

*Note that the classes No Gesture and Doing Other Things are obviously not mapped to any action.*

Briefly, the way the whole system works is as follows: videos are captured in our computer, sent to the Cloud where they are processed and classified, and then a response is returned and the computer performs the action contained on it. To manage the message sending in between the two machines, we used the message broker [RabbitMQ](https://www.rabbitmq.com/).

![alt text](https://github.com/gesturesAidl/video_processor/blob/main/images/webcam.jpg?raw=true)

#### VIDEO CAPTURING

To be consistent with the training data, the videos processed will be 3 seconds long and captured at 12 fps. If we just recorded videos every 3 seconds, we may encounter the problem that a gesture performance is split between two, causing problems with its identification. To address that problem we decided to overlap partially  the videos processed, specifically a 50% overlap. Hence, each frame is going to be used in two different videos. Also, to avoid unnecessarily sending the data twice, we capture 1.5 second videos and send them to the Cloud, where they are going to be joined appropriately.

#### VIDEO PROCESSING

Once the videos are received on Google Cloud, their Optical Flow is computed and the respective features are extracted from them, which go through the classification network. Then, a message is sent to our computer containing the class with the highest probability as well as the probability value.

#### ACTION EXECUTION

When the response message is received, if the probability value exceeds a certain threshold the corresponding action is executed. We use thresholds because as we are performing real actions on the computer, we want the model to be confident enough on the predictions.


## HOW TO RUN THE PROGRAM

### Installation

Before continuing, make sure you have Python installed and available from your command line.
You can check this by simply running:

```bash
#!/bin/bash
$ python --version
```

Your output should look similar to:  ```3.8.2.```
If you do not have Python, please install the latest 3.8 version from python.org.

#### Install Docker
Make sure you have docker installed in your computer by typing: 
```bash
$ docker -v
```
If installed, you should get an output similar to: `Docker version 20.10.3`. If you do not have Docker, install the latest version from its official site: https://docs.docker.com/get-docker/

#### Install docker-compose
Make sure you have docker-compose installed in your computer by typing: 
```bash
$ docker-compose -v
```
If installed, you should get an output similar to: `docker-compose version 1.27.4`. If you do not have docker-compose, install the latest version from its official site: https://docs.docker.com/compose/install/

#### Install Miniconda

To test your installation, in your terminal window or Anaconda Prompt, run the command: 
```bash
$ conda list
```
And you should obtain the list of packages installed in your base environment.

#### Create your Miniconda environment

>  Notice: This repository has been designed in order to allow being executed with two different environments. If you have a GPU in your computer, make use of the file "environment_gpu.yml" during the next section, in case you only have CPU, use the file "environment.yml" to prepare your environment.

Execute:

```bash
 $ conda env create -f environment_gpu.yml
```

This will generate the `videoprocgpu` environment with all the required tools and packages installed.

Once your environment had been created, activate it by typing:

```bash
$ conda activate videoprocgpu
```

#### Create your .env file
Create a folder with name `/env` inside the video_processor root directory folder and then, create a `.env` file inside it.
Copy the following code to your `.env` file:

    RABBIT_USER="..."
    RABBIT_PW="..."
    RABBIT_HOST="..."
    RABBIT_PORT="..."
    VIDEOS_OUT_PATH="..."
    MODEL_DIR="..."
    GPU="..."
    
Replace the dots in your `.env` file with the following information:
* RABBIT_USER: Default rabbit username.
* RABBIT_PW: Default rabbit user password.
* RABBIT_HOST: The IP or domain name where your rabbit service is hosted.
* RABBIT_PORT: Default port of your rabbit service.
* VIDEOS_OUT_PATH: Absolute path to the folder where you want to store your real time capturing videos. 
* MODEL_DIR: Absolute path to the `./models` folder. By default this directory is placed at the root of this repository.
* GPU: This field should be set to `True` in case you are using one GPU and `False` in case you are not.

### RUN the project
##### RabbitMQ
Activate the project service dependencies by starting a rabbitmq container. Go to the `/deployment` folder and edit the `docker-compose.yml` file in order to add your user and password credentials. 
* ${RABBIT_USER}: Replace by your rabbit default  user name.
* ${RABBIT_PW}: Replace by your rabbit default password. 

When your credentials had been set, from the same directory type:

```bash
$ docker-compose up rabbitmq
```

##### Video processor app
Finally, to start the video_processor `app`, place yourself in the repository root dir and type: 
```bash
$ python3 app/app.py
```
Now, the video processor app will be running as a process in your terminal and the server will be waiting for requests. You will see a log in your command line like the following:

    [X] Start Consuming:

    
