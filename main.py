from skimage import data as skdata
from skimage.transform import resize as skresize
import os
import numpy as np
import datetime

import keras
from keras import optimizers
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.utils import shuffle

IM_SIZE = 32
TRAIN_DATA_DIR = "BelgiumTSC_Training/Training"
TEST_DATA_DIR = "BelgiumTSC_Testing/Testing"
EPOCHS = 100

def load_data(data_dir):
    global IM_SIZE
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir)
                      if f.endswith(".ppm")]
        for f in file_names:
            im = skdata.imread(f)
            images.append(skresize(im, (IM_SIZE, IM_SIZE)).flatten())
            d = int(d)
            label = [0 if d != i else 1 for i in range(64)]
            labels.append(label)
    return images, labels

def create_fully_connected_model(input_dim, activation='relu', kernel_initializer='normal', learning_rate=0.0001):
    """ Creates and compiles fully connected model. """
    model = Sequential()
    model.add(Dense(256, input_dim=input_dim, activation=activation))
    model.add(Dense(128, activation=activation))
    model.add(Dense(64, kernel_initializer=kernel_initializer, activation='softmax'))
    # model.add(Dense(64, input_dim=input_dim, activation='softmax'))

    optimizer = optimizers.adam(lr=learning_rate)
    model.compile(optimizer=optimizer, metrics=['accuracy'], loss='categorical_crossentropy')
    return model

images, labels = load_data(TRAIN_DATA_DIR)
images, labels = shuffle(images, labels)
model = create_fully_connected_model(3072)
model.fit(np.array(images), np.array(labels), epochs=EPOCHS, validation_split=0.1,
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')])

# save model and data
name = 'models/classifier_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model.save(name + '.h5')
with open(name+'.txt', 'w') as f:
    f.write('EPOCHS: '+str(EPOCHS))

# read test data
print('\n\n\nTESTING:')
test_images, test_labels = load_data(TEST_DATA_DIR)
print(model.evaluate(np.array(test_images), np.array(test_labels)))
