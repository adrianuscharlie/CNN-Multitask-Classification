import tensorflow as tf
import numpy as np
import pandas as pd
import cv2 as cv
import os
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split

class Dataset:
    def __init__(self) -> None:
        self.path='./images/'
        self.dataset={'images':[],'emotion':[],'id':[],'gender':[],}
        self.loadImage()
        self.gender={value:index  for index,value in enumerate(set(self.dataset['gender']))}
        self.gender_idx={value:key for key,value in self.gender.items()}
        self.emotion={value:index  for index,value in enumerate(set(self.dataset['emotion']))}
        self.emotion_idx={value:key for key,value in self.emotion.items()}

    def loadImage(self):
        additional=pd.read_csv('./emotions.csv',index_col=0)
        additional.drop('country',inplace=True,axis=1)
        additional=additional.to_dict()
        for images in os.listdir(self.path):
            image_dir=os.path.join(self.path,images)
            for dir in os.listdir(image_dir):
                path=os.path.join(image_dir,dir)
                self.dataset['images'].append(path)
                self.dataset['emotion'].append(dir[:-4])
                self.dataset['id'].append(images)
                self.dataset['gender'].append(additional['gender'][int(images)])
        self.dataset=pd.DataFrame(self.dataset)
    
    def preprocessedImage(self,image,grayscale=False):
            """
            Tahapan preprocessing image:
            convert image dari rgb ke grayscale -> Equalize histogram -> Gausian Blurr -> Resize (224,224,1)->
            rescale -> 1/255.0
            """
            # Convert the image to grayscale
            image=cv.cvtColor(image,cv.COLOR_BGR2RGB)
            if grayscale:
                gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
                equalized= cv.equalizeHist(gray_image)
                blurred = cv.GaussianBlur(equalized, (5, 5), 0)
            # Resize the image to the target size
            resized_image = cv.resize(blurred if grayscale else image, (224,224))
            normalized = resized_image.astype('float32') / 255.0
            return normalized
    
    def getSample(self):
         random_idx=random.randint(0,len(self.dataset['images']))
         data=self.dataset.iloc[random_idx]
         image=cv.imread(data['images'])
         image=cv.cvtColor(image,cv.COLOR_BGR2RGB)
         gender=data['gender']
         emotion=data['emotion']
         plt.title(f'Sample Image {gender} person with {emotion} emotion')
         plt.axis('off')
         plt.imshow(image)
    
    def generateDataset(self,train_size):
        images=self.dataset['images'].apply(lambda x:self.preprocessedImage(cv.imread(x)))
        images=np.array(images.to_list())
        labels=np.array(list(zip(self.dataset['gender'].apply(lambda x:self.gender[x]),self.dataset['emotion'].apply(lambda x:self.emotion[x]))))
        x_train,x_test,y_train,y_test=train_test_split(images,labels,train_size=train_size,random_state=42)
        return x_train,x_test,y_train,y_test

from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
class CustomModel(tf.keras.Model):
    def __init__(self):
        # Define the input shape for the images
        input_shape = (224, 224, 3)

        # Load the VGG16 model (without the top classification layer)
        vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        for layer in vgg_model.layers:
            layer.trainable = False

        # Create the input layer for the images
        input_images = tf.keras.Input(shape=input_shape, name='input_images')

        # Connect the input to the VGG16 model
        x = vgg_model(input_images)

        # Flatten the output of the VGG16 layers
        x = layers.Flatten()(x)

        # Dense layers for gender classification
        x_gender = layers.Dense(128, activation='relu')(x)
        output_gender = layers.Dense(1, activation='sigmoid', name='output_gender')(x_gender)

        # Dense layers for emotion classification
        x_emotion = layers.Dense(128, activation='relu')(x)
        output_emotion = layers.Dense(8, activation='softmax', name='output_emotion')(x_emotion)
        # Create the multi-output model with input and output layers
        super().__init__(inputs=input_images, outputs=[output_gender, output_emotion])
        self.compile(optimizer='adam',
              loss={'output_gender': 'binary_crossentropy',
                    'output_emotion': 'sparse_categorical_crossentropy'},
              metrics={'output_gender': 'accuracy',
                       'output_emotion': 'accuracy'})
        self.summary()
    def trainModel(self,x_train,y_train):
        self.history=self.fit(x_train,{'output_gender':y_train[:,0],'output_emotion':y_train[:,1]},epochs=20)
    
    def evaluateModel(self,x_test,y_test):
        self.evaluate(x_test,{'output_gender':y_test[:,0],'output_emotion':y_test[:,1]})
    def predictImage(self,image):
        predicted=self.predict(np.expand_dims(image,axis=0))
        gender,emotion=0 if predicted[0]<0.5 else 1,np.argmax(predicted[1])
        return gender,emotion
    

    def preprocessImage(self,image):
        image=cv.cvtColor(image,cv.COLOR_BGR2RGB)
        resized_image = cv.resize(image, (224,224))
        normalized = resized_image.astype('float32') / 255.0
        return normalized

    def stream(self,gender_idx,emotion_idx):
        face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # define video capture -> load video dari webcam
        video_capture = cv.VideoCapture(0)
        while True:
            isTrue,frame=video_capture.read()
            preprocessed_frame=self.preprocessImage(frame)
            frame_expanded=tf.expand_dims(preprocessed_frame,axis=0)
            # make predictions
            predictions = self.predict(frame_expanded)
            gender,emotion=0 if predictions[0]<0.5 else 1,np.argmax(predictions[1])
            label_gender,label_emotion=gender_idx[gender],emotion_idx[emotion]
            value_gender=predictions[0].squeeze()*100 if label_gender=='FEMALE' else (1.0-predictions[0].squeeze())*100
            value_emotion=predictions[1].max()*100
            faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv.putText(frame,str(f'{value_gender:.2f}% {label_gender}'),org=(x-50, y-10),fontFace=cv.FONT_HERSHEY_SIMPLEX,fontScale=0.9,color=(255,0,0),thickness=2,lineType=cv.LINE_AA)
                cv.putText(frame,str(f'{value_emotion:.2f}% {label_emotion}'),org=(x-50, y-40),fontFace=cv.FONT_HERSHEY_SIMPLEX,fontScale=0.9,color=(255,0,0),thickness=2,lineType=cv.LINE_AA)
            cv.imshow('Video', frame)
            # Exit loop if 'q' is pressed
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
                # Destroy window when the real time predictions end
        cv.destroyAllWindows()