"""
University of California, Irvine
MSWE - 248P
Sherlin Mary Koshy (smkoshy)
"""
#Reference URL : https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/3.6-classifying-newswires.ipynb

#import needed libraries
import tensorflow.keras
import keras
from tensorflow.keras.datasets import reuters
from keras.utils.np_utils import to_categorical
from tensorflow.keras import models
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

#load training/test data and training/test labels
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
#print(len(train_data))
#print(len(test_data))
#print(train_data[10])
#print(train_labels[10])

#decode into words to verify
#word_index = reuters.get_word_index()
#reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
#decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
#print(decoded_newswire)

#method to vectorize the newswires data
def vectorize_newswires(news_reports, dimension=10000):
    results = np.zeros((len(news_reports), dimension))
    for i, word in enumerate(news_reports):
        results[i, word] = 1.
    return results

#vectorized training data
x_train = vectorize_newswires(train_data)
#vectorized test data
x_test = vectorize_newswires(test_data)

#vectorize training/test labels using keras built in one-hot-encoding
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

#create validation data and label sets from training data and labels
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

#building the initial network - 2 hidden layers 64 neurons each
#output layer of 46 neurons(1 per category of report) hidden layer activation-relu
#output layer activation-softmax(for prob. dist. across 46 categories) optimizer-rmsprop
#loss function-categorical_crossentropy and evaluation metric-accuracy

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
#model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#train model for 20 epochs validating against validation set in each epoch
"""
history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512,
                    validation_data=(x_val, y_val))

#plot graph of training vs validation loss
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#plot graph of training vs validation accuracy
plt.clf()
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
"""
#after above validation network appears to overfit at around 8 epochs (for initial network)
#train a new network for 8 epochs and evaluate against test set
#"""
print("***Retraining on new Network***")
model.fit(partial_x_train, partial_y_train, epochs=9, batch_size=512, #epoch 9 for best config of 1 layer 64 neurons
          validation_data=(x_val, y_val))
print("***Results on Test Set***")          
results = model.evaluate(x_test, one_hot_test_labels)
#"""