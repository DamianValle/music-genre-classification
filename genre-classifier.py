import tensorflow as tf
import keras
from keras import layers
from keras.layers import Activation, Dense, Dropout, Conv1D, Conv2D, Flatten, BatchNormalization, ZeroPadding2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling1D, AveragePooling2D, Input, Add
from keras.models import Sequential
from keras import regularizers
from keras.optimizers import SGD, Adam
import keras.backend as K
from keras.models import load_model
from keras.callbacks import EarlyStopping
import ffmpeg

#matplotlib inline
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display
import numpy as np
from numpy import argmax
import pandas as pd
import random
import warnings
import os
warnings.filterwarnings('ignore')

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

os.chdir('/zhome/12/f/134534/Desktop/genre-classifier')
os.environ["CUDA_VISIBLE_DEVICES"]="0"

###############################
#Dataloader and quick overview#
###############################
data = pd.read_csv('music_analysis.csv')
print('Shape of the data: {}'.format(data.shape))

def genre_number(i):
    if i == 'Hip-Hop':
        return 0
    elif i == 'Pop':
        return 1
    elif i == 'Folk':
        return 2
    elif i == 'Rock':
        return 3
    elif i == 'Experimental':
        return 4
    elif i == 'International':
        return 5
    elif i == 'Electronic':
        return 6
    else: #"Instrumental"
        return '7'

data['genre_number'] = data['genre'].apply(genre_number)
data['file_name'] = data['file_name'].apply(lambda x: '{0:0>6}'.format(x))
data['path'] = '/work3/s182091/fma_small/' + data['file_name'].str[:3] + '/' + data['file_name'] + ".wav"


dataset = []
i = 0
for row in data.itertuples():
    i = i + 1
    if(i%10==0):
        print(i)
    y, sr = librosa.load(row.path, duration=10)
    ps = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(ps, ref=np.max), y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()
    #plt.savefig('elspectro.png')

    if ps.shape != (128, 431): continue
    dataset.append((ps, row.genre_number))

print("Number of samples: ", len(dataset))

print("Shuffling dataset...")
random.shuffle(dataset)


train = dataset[:7000]
val = dataset[7001:7500]
test = dataset[7501:]

X_train, Y_train = zip(*train)
X_val, Y_val = zip(*val)
X_test, Y_test = zip(*test)

X_train = np.array([x.reshape( (128, 431, 1) ) for x in X_train])
X_val = np.array([x.reshape( (128, 431, 1) ) for x in X_val])
X_test = np.array([x.reshape( (128, 431, 1) ) for x in X_test])

Y_train = np.array(keras.utils.to_categorical(Y_train, 8))
Y_val = np.array(keras.utils.to_categorical(Y_val, 8))
Y_test = np.array(keras.utils.to_categorical(Y_test, 8))

#####################
# Model architecure #
#####################

model = Sequential()
input_shape=(128, 431, 1)

model.add(Conv2D(32, (5, 5), strides=(1, 1), input_shape=input_shape))
model.add(MaxPooling2D((2, 2), strides=(2,2)))
model.add(Activation('relu'))

model.add(Conv2D(32, (5, 5), padding="same"))
model.add(MaxPooling2D((2, 2), strides=(2,2)))
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3), padding="same"))
model.add(MaxPooling2D((2, 2), strides=(2,2)))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dropout(rate=0.5))

model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(rate=0.5))

model.add(Dense(8))
model.add(Activation('softmax'))

model.summary()

##################
# Model training #
##################

#epochs = 200
#batch_size = 32
#learning_rate = 0.01
#decay_rate = learning_rate / epochs
#momentum = 0.9
#sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
#model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=['accuracy'])

epochs = 200
batch_size = 32

adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=['accuracy'])

history = model.fit(X_train, Y_train, batch_size = batch_size, nb_epoch=epochs,
          validation_data = (X_val, Y_val))

score = model.evaluate(x=X_test, y=Y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

plt.figure(figsize=(12,8))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['acc','val_acc'])
plt.savefig("acc.png")

plt.figure(figsize=(12,8))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss','val_loss'])
plt.savefig("loss.png")

plt.figure(figsize=(12,8))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.xlabel('epoch')
plt.legend(['loss','val_loss', 'acc','val_acc'])
plt.savefig('acc-loss.png')

model.save('genre_classifier_12may.h5')
