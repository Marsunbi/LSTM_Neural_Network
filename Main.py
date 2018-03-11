import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

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

