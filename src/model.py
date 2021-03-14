import pickle
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Note: Use LSTM if you dont have a NVDIA  else use CuDNNLSTM
from tensorflow.python.keras.layers import LSTM, CuDNNLSTM, Dense, Dropout, GlobalMaxPool1D, Bidirectional, Conv1D, MaxPooling1D
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python import keras
#import keras

# Set training logger
logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

# Reading the tokenizer pickle file
with open(r"dataset/word_tokenizer.pickle", "rb") as input_file:
    tokenizer = pickle.load(input_file)

# Constants for model configuration
GPU = False
MAX_WORDS = 80
NO_OF_CLASSES = 5
VOCAB_SIZE = 10000
EPOCHS = 200
BATCH_SIZE = 5

# Loading the dataset
dataset = np.load('dataset/preprocessed_dataset.npy')

# Spliting into X and y
X = dataset[:, 0:80]
y = dataset[:, 80]
y = pd.get_dummies(y).values

# Saving memory
del dataset

# Spliting the dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2, 
                                                    random_state=1969)
# Saving memory
del X, y

# Making the model
model = Sequential()
model.add(Embedding(VOCAB_SIZE, 128, input_length=MAX_WORDS))

model.add(Dropout(0.95))

model.add(Conv1D(filters=256, kernel_size=1, activation='relu'))
model.add(MaxPooling1D(3))

model.add(Conv1D(filters=512, kernel_size=1, activation='relu'))
model.add(MaxPooling1D(3))

model.add(Dropout(0.95))

if GPU:
    model.add(Bidirectional(CuDNNLSTM(256, return_sequences = True)))
else:
    model.add(Bidirectional(LSTM(256, return_sequences = True)))
    
model.add(GlobalMaxPool1D())

model.add(Dense(256, activation="relu"))
model.add(Dense(128, activation="relu"))

model.add(Dropout(0.95))

model.add(Dense(NO_OF_CLASSES, activation='softmax'))

# Compiling the model
#opt = keras.optimizers.Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

# Printing the model summary
print(model.summary())

# Added early stopping system to monnitor validation loss on each epoch and stops training when validation loss start to increase
monitor = EarlyStopping(monitor='val_loss', 
                        patience=EPOCHS, 
                        mode='min',
                        restore_best_weights=True)

# Saving the model in every epochs for some experiments
checkpoint = ModelCheckpoint(filepath="weights/model.{epoch:02d}-{val_loss:.2f}.h5")

# Starting the training process
model.fit(X_train, 
          y_train, 
          validation_data=(X_test, y_test), 
          epochs=EPOCHS, 
          batch_size=BATCH_SIZE, 
          verbose=1,
          callbacks=[monitor, checkpoint, tensorboard_callback])

# Saving the model
model.save('weights/model_best.h5')

# Use tensorboard --logdir logs/scalars to check the training curves