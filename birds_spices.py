# -*- coding: utf-8 -*-
"""birds_spices.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zNTPpGD-WMozLJNswp4_oHJd9AVL3hkI
"""

# from google.colab import drive
# drive.mount('/content/drive/')

# import the libraries as shown below

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

dataset_path = "/content/drive/MyDrive/notebook/datasets"

# re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path =dataset_path +'/train'
valid_path = dataset_path +'/val'
test_path = dataset_path + '/test'

# Import the Vgg 16 library as shown below and add preprocessing layer to the front of VGG
# Here we will be using imagenet weights

inception = MobileNetV2(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# don't train existing weights
for layer in inception.layers:
    layer.trainable = False

# useful for getting number of output classes
folders = glob(train_path +'/*')

# our layers - you can add more if you want
x = Flatten()(inception.output)

prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=inception.input, outputs=prediction)


# view the structure of the model
model.summary()

# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

# Use the Image Data Generator to import the images from the dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

val_datagen = ImageDataGenerator(rescale = 1./255)

# Make sure you provide the same target size as initialied for the image size
train_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

val_set = val_datagen.flow_from_directory(valid_path,
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')

# fit the model
# Run the cell. It will take some time to execute
r = model.fit_generator(
  train_set,
  validation_data=val_set,
  epochs=20,
  steps_per_epoch=len(train_set),
  validation_steps=len(val_set)
)

# plot the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

# save it as a h5 file in inversion v3 model


from tensorflow.keras.models import load_model

model.save('mobilenetv2.h5')

#load the model

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model=load_model('mobilenetv2.h5')

ref = dict(zip(list(train_set.class_indices.values()),list(train_set.class_indices.keys())))

def prediction(path):
  img = image.load_img(path, target_size=(224, 224))
  i = image.img_to_array(img)
  i = np.expand_dims(i, axis=0)
  img = preprocess_input(i)
  pred = np.argmax(model.predict(img), axis=1)
  print(f"the image belongs to {ref[pred[0]]}")

path = "datasets/test/DARJEELING WOODPECKER/3.jpg"
prediction(path)

train_set.class_indices

# modeling

#load the model
# from tensorflow.keras.applications.inception_v3 import preprocess_input

# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# model=load_model('mobilenetv2.h5')
# from PIL import Image

# def prediction(path):
#     img = image.load_img(path, target_size=(224, 224))
#     i = image.img_to_array(img)
#     import numpy as np
#     i = np.expand_dims(i, axis=0)
#     img = preprocess_input(i)
#     pred = np.argmax(model.predict(img), axis=1)

#     print(pred)

# path = r"/content/drive/MyDrive/notebook/datasets/test/FAIRY BLUEBIRD/3.jpg"
# prediction(path)