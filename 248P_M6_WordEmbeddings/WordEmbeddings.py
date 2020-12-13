"""
University of California, Irvine
MSWE - 248P
Sherlin Mary Koshy (smkoshy)
"""
#Reference URL :https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/6.1-using-word-embeddings.ipynb

#import necessary libraries
import tensorflow.keras
from tensorflow.keras.layers import Embedding
from tensorflow.keras.datasets import imdb
from tensorflow.keras import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

# Number of words to consider as features (10000 most common words)
max_features = 10000
# Reduce reviews to length of 20
maxlen = 20

# Load the data as lists of integers.
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# convert lists to 2D tensors of shape(samples,maxlen)
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

#build model with first layer as Embedding produces tensor with shape(10000, 8,maxlen)
#each word represented by 8 dimensional embedding, flatten to prepare for dense layer
#Binary classification hence activation sigmoid with a single neuron optimizer-rmsprop
#loss function - binary crossentropy and metric accuracy
model = Sequential()
model.add(Embedding(10000, 8, input_length=maxlen))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

print(model.summary())

#fit data to model embedding layers learn representations from input
history = model.fit(x_train, y_train, epochs=10, batch_size=32,validation_split=0.2)
print("*******Test Results****")
model.evaluate(x_test,y_test)