import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import h5py
import sys

#Load ASCII text and convert to lower-case
filename = 'wonderland.txt'
raw_text = open(filename).read()
raw_text = raw_text.lower()

#Create map of each unique character to int
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

number_of_characters = len(raw_text)
vocabulary = len(chars)

print("The number of characters: ", number_of_characters)
print("The number of vocals: ", vocabulary)

seq_length = 100
dataX = []
dataY = []

for i in range(0, number_of_characters - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
number_of_patterns = len(dataX)
print("Number of patterns: ", number_of_patterns)

X = np.reshape(dataX, (number_of_patterns, seq_length, 1))
X = X / float(vocabulary)
y = np_utils.to_categorical(dataY)

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

#filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
filepath="weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(X, y, epochs=50, batch_size=64, callbacks=callbacks_list)


'''
filename = "FILENAME"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

int_to_char = dict((i, c) for i, c in enumerate(chars))

start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print("Seed: ")
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
for i in range(1500):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(vocabulary)
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1: len(pattern)]
print("\Done")

'''