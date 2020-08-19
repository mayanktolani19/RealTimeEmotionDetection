import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import utils
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model

from IPython.display import SVG, Image
from livelossplot.tf_keras import PlotLossesCallback
import tensorflow as tf
print("Tensorflow Version: ", tf.__version__) 

img_size = 48
batch_size = 64

datagen_train = ImageDataGenerator(horizontal_flip=True)
train_generator = datagen_train.flow_from_directory("train/",target_size=(img_size,img_size),color_mode='grayscale',batch_size=batch_size,class_mode="categorical",shuffle=True)

datagen_validation = ImageDataGenerator(horizontal_flip=True)
validation_generator = datagen_validation.flow_from_directory("test/",target_size=(img_size,img_size),color_mode='grayscale',batch_size=batch_size,class_mode="categorical",shuffle=True)

model = Sequential()

# CONV - 1
# 64 filters each of 3*3, pad so that output size is same as input size
model.add(Conv2D(64, (3,3), padding='Same', input_shape=(img_size,img_size,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# CONV - 2
# 128 filters each of 5*5, pad so that output size is same as input size
model.add(Conv2D(128, (5,5), padding='Same', input_shape=(img_size,img_size,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# CONV - 3
# 512 filters each of 3*3, pad so that output size is same as input size
model.add(Conv2D(512, (3,3), padding='Same', input_shape=(img_size,img_size,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# CONV - 4
# 512 filters each of 3*3, pad so that output size is same as input size
model.add(Conv2D(512, (3,3), padding='Same', input_shape=(img_size,img_size,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

# Fully connected layer 1
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

# Fully connected layer 2
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(7, activation='softmax'))

opt = Adam(lr=0.0005)
model.compile(optimizer = opt, loss='categorical_crossentropy', metrics = ['accuracy'])
model.summary()

epochs =15
steps_per_epoch = train_generator.n//train_generator.batch_size
validation_steps = validation_generator.n//validation_generator.batch_size

checkpoint = ModelCheckpoint("model_weights.h5",monitor="val_accuracy",save_weights_only=True, mode="max",verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=2, min_lr=0.00001, mode='auto')
callback = [PlotLossesCallback(),checkpoint,reduce_lr]
history = model.fit(x=train_generator,steps_per_epoch=steps_per_epoch,epochs=epochs,validation_data = validation_generator, validation_steps=validation_steps,callbacks=callback)

model_json = model.to_json()
with open("model.json","w") as json_file:
    json_file.write(model_json)