"""
University of California, Irvine
MSWE - 248P
Sherlin Mary Koshy (smkoshy)
"""
#Reference URL : https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/3.5-classifying-movie-reviews.ipynb

#import needed libraries
import tensorflow.keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import numpy as np
import matplotlib.pyplot as plt

#load train and test datasets, verify by printing values
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)
#print(train_data[0])
#print(train_labels[0])
#print(max([max(review) for review in train_data]))

#decode review into English words to verify
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]]) 
#print(decoded_review)

#method to vectorize data into tensors
def vectorize_reviews(reviews, dimension = 10000):
    results = np.zeros((len(reviews), dimension))
    for i, review in enumerate(reviews):
        results[i, review] = 1
    return results

#vectorize training and test data/training and test labels
x_train = vectorize_reviews(train_data)
x_test = vectorize_reviews(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
#print(x_train[0])
#print(y_train[:10])

#build network - 2  dense hidden layers, 16 neurons per layer, 
#hidden layer activation-relu, ouput layer activaiton-sigmoid
#optimizer-rmsprop, loss function-binary crossentropy, evaluation metric-accuracy
model = models.Sequential()
model.add(layers.Dense(16, activation = 'relu', input_shape = (10000,)))
#model.add(layers.Dense(16, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.compile(optimizer = optimizers.RMSprop(lr=0.01), loss = 'binary_crossentropy', 
metrics = ['accuracy'])

#validate network - split train data and labels into train and validatation sets
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

#fit training data to network(train) for 20 epochs in mini batches of size 512 
#monitor accuracy against performance on validation set
"""
history = model.fit(partial_x_train, partial_y_train, epochs = 20, batch_size = 512,
validation_data=(x_val, y_val))

#using values in History object returned by fit() plot graph of performance trend
history_dict = history.history
#print(history_dict.keys())
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs= range(1, len(acc)+1)
#plot graph of training and validation loss
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#plot graph of training and validation accuracy
plt.clf()
plt.plot(epochs, acc, 'bo', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
"""
#after above training and validation, graphs showed overfit began to occur at 4 epochs (for initial network)
#train new model to run for only 4 epochs on the entire training set
#"""
model.fit(x_train, y_train, epochs=2, batch_size=512) #2 epoch is for best config - 1 hidden 16 neurons relu
print("***Predictions***")
model.predict(x_test)
results = model.evaluate(x_test, y_test)
print("***results***")
print(results)
#"""