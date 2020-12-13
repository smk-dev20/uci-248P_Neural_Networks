"""
University of California, Irvine
MSWE - 248P
Sherlin Mary Koshy (smkoshy)
"""
#Reference URL :https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/5.1-introduction-to-convnets.ipynb

#import needed libraries
import tensorflow.keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

#load data from mnist 60000 training and 10000 test images
#and scale so that training values are between 0 and 1, one-hot-encode labels
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#define model 3 conv2D layers with channels 32,64 and 64 sliding window(3,3)
#two maxpool2D layers of window (2,2)
#activation-relu, grayscale input image(28*28*1)
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#Flatten 3D output of conv2D to 1D add Dense classifier layer with 64 neurons activation relu
#Output dense layer of 10 neurons with activation-sigmoid for prob distribution across 10 classes
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
#print(model.summary())

#compile and train model for 5 epochs, loss function- categorical crossentropy
#optimizer-rmsprop, evaluation metric-accuracy
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)

#evaluate model on test data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("***Test Accuracy***")
print(test_acc)