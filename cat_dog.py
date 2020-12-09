#Import the libraries
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing import image
import numpy as np
from keras.models import load_model

import tkinter as tk
from tkinter import filedialog
from tkinter import *

'''
#For the training we will need to have a large number of pictures of both the dogs and the cats. 
#But I am using a trained version of the model.

#ImageDataGenerator performs resizing, rotating the images and many more functions.
#TO make your model more robust

from keras.preprocessing.image import ImageDataGenerator
#Image Preprocessing
image_gen = ImageDataGenerator(rotation_range=30, # rotate the image 30 degrees
                               width_shift_range=0.1, # Stretch the images or shift the pic width by a max of 10%
                               height_shift_range=0.1, # Shift the pic height by a max of 10%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.2, # Shear means cutting away part of the image (max 20%)
                               zoom_range=0.2, # Zoom in by 20% max
                               horizontal_flip=True, # Allo horizontal flipping
                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value
                              )

#Generating the manipulated images.
image_gen.flow_from_directory('CATS_DOGS/train')

#Test Images
image_gen.flow_from_directory('CATS_DOGS/test')

image_shape = (150,150,3)

#Building the Model
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D

#Create the Model.
model = Sequential()

#Convolutional Layer
model.add(Conv2D(filters=32, kernel_size=(3,3),input_shape=(150,150,3), activation='relu',))
#MaxPooling Layer
model.add(MaxPooling2D(pool_size=(2, 2)))

#Convolutional Layer
model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=(150,150,3), activation='relu',))
#MaxPooling Layer
model.add(MaxPooling2D(pool_size=(2, 2)))

#Convolutional Layer
model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=(150,150,3), activation='relu',))
#MaxPooling Layer
model.add(MaxPooling2D(pool_size=(2, 2)))

#Flatten Layer
model.add(Flatten())

#Add A Dense Layer
model.add(Dense(128))
model.add(Activation('relu'))

# Dropouts help reduce overfitting by randomly turning neurons off during training.
# Here we say randomly turn off 50% of neurons.
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

batch_size = 16

train_image_gen = image_gen.flow_from_directory('CATS_DOGS/train',
                                               target_size=image_shape[:2],
                                               batch_size=batch_size,
                                               class_mode='binary')
test_image_gen = image_gen.flow_from_directory('CATS_DOGS/test',
                                               target_size=image_shape[:2],
                                               batch_size=batch_size,
                                               class_mode='binary')

#Training
results = model.fit_generator(train_image_gen,epochs=1,
                              steps_per_epoch=150,
                              validation_data=test_image_gen,
                             validation_steps=12)
'''

root = tk.Tk()

root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
print(root.filename)
root.withdraw()

new_model = load_model('cat_dog_100epochs.h5')


input_file = str(root.filename)

#Image Preprocessing
input_image = image.load_img(input_file, target_size = (150,150))
input_image = image.img_to_array(input_image)
input_image = np.expand_dims(input_image, axis = 0)
input_image = input_image/255

prediction = new_model.predict_classes(input_image)

cat = cv2.imread('cat.jpg',0)
#cat = cv2.cvtColor(cat, cv2.COLOR_BGR2RGB)

dog = cv2.imread('dog.jpg',0)
#dog = cv2.cvtColor(dog, cv2.COLOR_BGR2RGB)

if prediction[0][0] == 0:
    while True:

        cv2.imshow('Cat Window', cat)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
else:
     while True:

        cv2.imshow('Dog Window', dog)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
     cv2.destroyAllWindows()
