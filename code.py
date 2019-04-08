import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split

import pandas as pd

from keras.layers import Dropout, Dense, GlobalAveragePooling2D
from keras.layers import Flatten, MaxPooling2D, Conv2D
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg19 import VGG19
from keras.applications.nasnet import NASNetLarge
from keras.applications.xception import Xception
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.callbacks import CSVLogger
from keras.models import Model
import keras 

from keras.preprocessing.image import ImageDataGenerator

x_train = np.load('../X_train.npy')
x_val = np.load('../X_val.npy')
y_train = np.load('../y_train.npy')
y_val = np.load('../y_val.npy')



# y_train_ohe = pd.get_dummies(y_train.reset_index(drop=True)).as_matrix()
# y_val_ohe = pd.get_dummies(y_val.reset_index(drop=True)).as_matrix()
BATCH_SIZE = 64

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=30,
                                   width_shift_range=0.2, height_shift_range=0.2,
                                   horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)
val_generator = val_datagen.flow(x_val, y_val, batch_size=BATCH_SIZE)


def transfer_model(num_classes, input_shape):
    base_model = Xception(weights='imagenet', include_top=False,
                       input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[96:]:
        layer.trainable = True
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def simple_cov_model(num_classes, input_shape):
    model = keras.Sequential()

    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))

def build_model(num_classes, input_shape, is_transfer=False):
    if is_transfer:
        return transfer_model(num_classes, input_shape)
    else:
        return simple_cov_model(num_classes, input_shape)


NUM_CLASSES = 50
INPUT_SHAPE = (331, 331, 3)
model = build_model(NUM_CLASSES, INPUT_SHAPE, is_transfer=True)
    
model.load_weights('../models_saved/weights-improvement-23-1.9502-bigger.hdf5')

opt = keras.optimizers.Adam(lr=1e-4)
model.compile(
    optimizer = opt,
    loss = 'sparse_categorical_crossentropy',
    metrics=['accuracy'])

# Directory where the checkpoints will be saved

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath='../models_saved/weights-improvement-{epoch:02d}-{val_loss:.4f}-bigger.hdf5',
    save_weights_only=True,
    save_best_only=True,
    monitor='val_loss')

csv_logger = CSVLogger('../training_log.csv')

tensorboard = keras.callbacks.TensorBoard(log_dir='../logs')

STEPS_PER_EPOCH = x_train.shape[0] // BATCH_SIZE
VALIDATION_STEPS = x_val.shape[0] // BATCH_SIZE

model.summary()

# reduce = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, mode='auto', verbose=1)
early = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=5, mode='auto', verbose=1)

history = model.fit_generator(train_generator, epochs=50, steps_per_epoch=STEPS_PER_EPOCH,
                              callbacks=[checkpoint_callback, early], 
                              validation_data=val_generator, verbose=1, validation_steps=VALIDATION_STEPS)


