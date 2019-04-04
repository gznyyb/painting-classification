#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 15:21:29 2019

@author: xianghongluo
"""

import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split

from keras.layers import Dropout, Dense, GlobalAveragePooling2D
from keras.layers import Flatten, MaxPooling2D, Conv2D
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import CSVLogger
from keras.models import Model
import keras 

from keras.preprocessing.image import ImageDataGenerator

np.random.seed(10)

# tf.enable_eager_execution()
# print("Eager execution: {}".format(tf.executing_eagerly()))

with open('labels_list', 'rb') as fp:
    labels = pickle.load(fp)

image_data = np.load('image_data.npy')

unique_labels = sorted(set(labels))
name2idx = {label: index for index, label in enumerate(unique_labels)}
idx2name = np.array(unique_labels)

int_labels = [name2idx[named_label] for named_label in labels]


X_train, X_test, y_train, y_test = train_test_split(image_data, int_labels,
                                                    test_size=0.25,
                                                    random_state=10,
                                                    stratify=int_labels)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                  test_size=0.25,
                                                  random_state=10,
                                                  stratify=y_train)


# def train_data_generator():
#     train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50,
#                                        width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
#                                        horizontal_flip=True, fill_mode='nearest')
#     train_data_generate = train_datagen.flow(X_train, y_train,
#                                              batch_size=BATCH_SIZE)
#     for image, label in train_data_generate:
#         yield image, label
# 
# 
# def val_data_generator():
#     val_datagen = ImageDataGenerator(rescale=1./255)
#     val_data_generate = val_datagen.flow(X_val, y_val,
#                                          batch_size=BATCH_SIZE)
#     for image, label in val_data_generate:
#         yield image, label


BATCH_SIZE = 32
# BUFFER_SIZE = 10000

# train_data_set = tf.data.Dataset.from_tensor_slices((X_train, y_train))
# val_data_set = tf.data.Dataset.from_tensor_slices((X_val, y_val))
# 
# train_data_set = tf.data.Dataset.from_generator(train_data_generator,
#                                                 (tf.float32, tf.int32),
#                                                 ((BATCH_SIZE, 224, 224, 3), (64,)))
# val_data_set = tf.data.Dataset.from_generator(val_data_generator,
#                                               (tf.float32, tf.int32),
#                                               ((BATCH_SIZE, 224, 224, 3), (64,)))
# 
# train_data_set_shuffled = train_data_set.shuffle(BUFFER_SIZE)
# val_data_set_shuffled = val_data_set.shuffle(BUFFER_SIZE)


# def train_image_transformation(x, y):
#     datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50,
#                                  width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
#                                  horizontal_flip=True, fill_mode='nearest')
#     return datagen.random_transform(x, seed=10), y
# 
# 
# def val_image_transformation(x, y):
#     return x / 25, y
# 
# train_data_set = train_data_set.map(train_image_transformation)
# val_data_set = val_data_set.map(val_image_transformation)
# 
# train_data_set_shuffled = train_data_set.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
# val_data_set_shuffled = val_data_set.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=30,
                                   width_shift_range=0.2, height_shift_range=0.2,
                                   horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
val_generator = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE)


def transfer_model(num_classes):
    base_model = InceptionV3(weights='imagenet', include_top=False)
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
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
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def build_model(num_classes, input_shape, is_transfer=False):
    if is_transfer:
        return transfer_model(num_classes)
    else:
        return simple_cov_model(num_classes, input_shape)


NUM_CLASSES = len(name2idx.keys())
INPUT_SHAPE = (224, 224, 3)
model = build_model(NUM_CLASSES, INPUT_SHAPE, is_transfer=True)

# old_checkpoint_dir = './training_checkpoints'

# model.load_weights('weights-improvement-178-1.3679-bigger.hdf5')

opt = keras.optimizers.Adam()
model.compile(
    optimizer = opt,
    loss = 'sparse_categorical_crossentropy',
    metrics=['accuracy'])

# Directory where the checkpoints will be saved

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath='models_saved\\weights-improvement-{epoch:02d}-{val_loss:.4f}-bigger.hdf5',
    save_weights_only=True,
    save_best_only=True,
    monitor='val_loss')

csv_logger = CSVLogger('training_log.csv')

tensorboard = keras.callbacks.TensorBoard(log_dir='./logs')

STEPS_PER_EPOCH = X_train.shape[0] // BATCH_SIZE
VALIDATION_STEPS = X_val.shape[0] // BATCH_SIZE

# model.summary()

reduce = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, mode='auto', verbose=1)
early = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, mode='auto', verbose=1)

history = model.fit_generator(train_generator, epochs=200, steps_per_epoch=STEPS_PER_EPOCH,
                              callbacks=[checkpoint_callback, csv_logger, tensorboard, reduce], 
                              validation_data=val_generator, verbose=1, validation_steps=VALIDATION_STEPS)



# layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
# pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
