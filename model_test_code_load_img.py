import pandas as pd
import numpy as np

data_labels = pd.read_csv('../labels/labels.csv')
target_labels = data_labels['breed']

train_folder = '../train/'
data_labels['image_path'] = data_labels.apply(lambda row: (train_folder + row["id"] + ".jpg" ), 
                                              axis=1)

from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array, load_img

# load dataset
train_data = np.array([img_to_array(load_img(img, target_size=(299, 299)))
                           for img in data_labels['image_path'].values.tolist()
                      ]).astype('float32')

unique_labels = sorted(set(target_labels))
name2idx = {label: index for index, label in enumerate(unique_labels)}
idx2name = np.array(unique_labels)
int_labels = [name2idx[named_label] for named_label in target_labels]

# create train and test datasets
x_train, x_test, y_train, y_test = train_test_split(train_data, int_labels, 
                                                    test_size=0.3, 
                                                    stratify=np.array(int_labels), 
                                                    random_state=42)

# create train and validation datasets
X_train, X_val, y_train, y_val = train_test_split(x_train, y_train, 
                                                  test_size=0.15, 
                                                  stratify=np.array(y_train), 
                                                  random_state=42)

np.save('../X_train.npy', arr=X_train)
np.save('../X_val.npy', arr=X_val)
np.save('../y_train.npy', arr=y_train)
np.save('../y_val.npy', arr=y_val)