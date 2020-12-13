"""
University of California, Irvine
MSWE - 248P
Sherlin Mary Koshy (smkoshy)
"""
#Reference URL :https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/6.4-sequence-processing-with-convnets.ipynb

#import needed libraries
import tensorflow.keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt


def preprocess_data():
    max_features = 10000  # top 10000 words in imdb
    maxlen = 500  # restrict words in each review to 500

    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')
    #pad each review to be only 500 words in length
    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('input_train shape:', x_train.shape)
    print('input_test shape:', x_test.shape)

    return x_train, y_train, x_test, y_test

def build_model():
    max_features = 10000  # top 10000 words in imdb
    maxlen = 500  # restrict words in each review to 500
    
    #model contains 2 1D convnets with 32 channels output, sliding window 7 and activation relu
    #1 maxpool layer with window size 5
    #globalmaxpool1D layer to flatten 3D tensor to 2D before classifier
    #1 neuron output dense layer for sentiment prediction
    #loss function - binary crossentropy otimizer-rmsprop and evaluation metric accuracy
    model = Sequential()
    model.add(layers.Embedding(max_features, 128, input_length=maxlen))
    model.add(layers.Conv1D(32, 7, activation='relu'))
    model.add(layers.MaxPooling1D(5))
    model.add(layers.Conv1D(32, 7, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(1))
    model.compile(optimizer=RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['acc'])   
    return model

def plot(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

def main():
    #load and preprocess training and test data from imdb
    x_train, y_train, x_test, y_test = preprocess_data()
    
    #Build model with 1DConvNet
    model = build_model()
    model.summary()
    #train and evaluate
    print(' main input_train shape:', x_train.shape)
    print('main input_test shape:', x_test.shape)
    history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
    plot(history)
    print("****Test Results***")
    model.evaluate(x_test, y_test)

if __name__ == "__main__":
    main()     