import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split

import pandas as pd

from keras.layers import Dropout, Dense, GlobalAveragePooling2D
from keras.layers import Flatten, MaxPooling2D, Conv2D
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import CSVLogger
from keras.models import Model
import keras 

from keras.preprocessing.image import ImageDataGenerator

x_train = np.load('X_train.npy')
x_val = np.load('X_val.npy')
y_train = np.load('y_train.npy')
y_val = np.load('y_val.npy')



# y_train_ohe = pd.get_dummies(y_train.reset_index(drop=True)).as_matrix()
# y_val_ohe = pd.get_dummies(y_val.reset_index(drop=True)).as_matrix()


BATCH_SIZE = 32

# Create train generator.
train_datagen = ImageDataGenerator(rescale=1./255, 
                                   rotation_range=30, 
                                   width_shift_range=0.2,
                                   height_shift_range=0.2, 
                                   horizontal_flip = 'true')
train_generator = train_datagen.flow(x_train, y_train, shuffle=False, 
                                     batch_size=BATCH_SIZE, seed=1)
                                     
# Create validation generator
val_datagen = ImageDataGenerator(rescale = 1./255)
val_generator = train_datagen.flow(x_val, y_val, shuffle=False, 
                                   batch_size=BATCH_SIZE, seed=1)         

from keras.optimizers import Adam

# Get the InceptionV3 model so we can do transfer learning
base_inception = InceptionV3(weights='imagenet', include_top=False, 
                             input_shape=(299, 299, 3))
                             
# Add a global spatial average pooling layer
out = base_inception.output
out = GlobalAveragePooling2D()(out)
out = Dense(512, activation='relu')(out)
out = Dense(512, activation='relu')(out)
total_classes = 50
predictions = Dense(total_classes, activation='softmax')(out)

model = Model(inputs=base_inception.input, outputs=predictions)

# only if we want to freeze layers
for layer in base_inception.layers:
    layer.trainable = False
    
# Compile 
model.compile(Adam(lr=.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
# model.summary()

# Train the model
batch_size = BATCH_SIZE
train_steps_per_epoch = x_train.shape[0] // batch_size
val_steps_per_epoch = x_val.shape[0] // batch_size

history = model.fit_generator(train_generator,
                              steps_per_epoch=train_steps_per_epoch,
                              validation_data=val_generator,
                              validation_steps=val_steps_per_epoch,
                              epochs=15, verbose=1)

