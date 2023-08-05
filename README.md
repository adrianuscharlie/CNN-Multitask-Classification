# Multitask Classification (Gender and Emotion)
 
Create Deep Convulutional Neural Network model to do multitask classification. This model can classify gender and emotion of people from picture or video straming. 
This model using pre-trained model VGG16 for the baseline, and add some modification in last layer to do multitask classification. 

<br>


## Features
- Tools
- Dataset
- Modelling
- Real Time Predictions

## Tools
- Python
- Tensorflow
- Opencv

## Dataset
The dataset is collected from Kaggle.com Through [this link](https://www.kaggle.com/datasets/tapakah68/facial-emotion-recognition). The dataset
contain folder contains images and a csv file for detail data like age, gender, country. Each folder, contains one person with 8 different expression
like happy, sad, disgust, angry, etc. From this dataset, the model will be learn how to classify gender and emotion from person image.


## Modelling
VGG16 are pre-trained model that commonly used to extract the image features. After that, the image will be pass through the flatten layer and
the output from flatten layer will be pass through for the input in each classification output layer.

![image](https://github.com/adrianuscharlie/CNN-Multitask-Classification/assets/72659267/2f6edcea-96ca-4806-8410-53d27a76b00c)


## Realtime Predictions
This is some example of real time predictions using the current model that already trained from previous step.

![image](https://github.com/adrianuscharlie/CNN-Multitask-Classification/assets/72659267/ded5ff9b-b577-4ae9-a849-df9a61c9b939)
![image](https://github.com/adrianuscharlie/CNN-Multitask-Classification/assets/72659267/c3b46083-7058-4514-8360-b7807877cc86)

