#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 09:40:33 2019

@author: xianghongluo
"""

# import packages for processing images
from PIL import Image
from tensorflow.keras.preprocessing import image as kp_image

# import other useful packages
import os
import numpy as np
import glob
import pickle
from sklearn.model_selection import train_test_split

# path to all the images
image_dir_path = '../best-artworks-of-all-time/resized'

# array to store the images and their respective labels 
image_data = []
labels = []

# loop to process all images and store the images and labels
for filename in glob.glob(os.path.join(image_dir_path,'*.jpg')):
    # get the label
    img_str_list = filename.split('\\')[-1].split('_')
    if len(img_str_list) == 2:
        labels.append(img_str_list[0])
    else:
        labels.append(' '.join(img_str_list[:2]))

    # process the image
    img = Image.open(filename) # open the image file
    img = img.resize((331, 331), Image.ANTIALIAS) # resize the image
    img = kp_image.img_to_array(img) # convert it to array 
    # copy the black and white images across color channel such that their dimensions
    # match the colored images'
    if img.shape[-1] != 3:
        img = np.repeat(img, 3, axis=2)
    img = np.expand_dims(img, axis=0)
    image_data.append(img) # store the image arrays 

# convert the list of image arrays into a big array
image_data = np.concatenate(image_data, axis=0)

# convert the string labels into numeric labels
unique_labels = sorted(set(labels))
name2idx = {label: index for index, label in enumerate(unique_labels)}
idx2name = np.array(unique_labels)
int_labels = [name2idx[named_label] for named_label in labels]

# split the data into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(image_data, int_labels,
                                                    test_size=0.25,
                                                    random_state=10,
                                                    stratify=int_labels)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                  test_size=0.25,
                                                  random_state=10,
                                                  stratify=y_train)

# save the training, validation, and test sets to disk 
np.save('../X_train.npy', arr=X_train)
np.save('../X_val.npy', arr=X_val)
np.save('../X_test.npy', arr=X_test)
np.save('../y_train.npy', arr=y_train)
np.save('../y_val.npy', arr=y_val)
np.save('../y_test.npy', arr=y_test)

# save the labels to disk
with open('../name2idx.pickle', 'wb') as handle:
    pickle.dump(name2idx, handle)

with open('../idx2name.pickle', 'wb') as handle:
    pickle.dump(idx2name, handle)

