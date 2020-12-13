"""
University of California, Irvine
MSWE - 248P
Sherlin Mary Koshy (smkoshy)
"""
#Reference URL :https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/6.2-understanding-recurrent-neural-networks.ipynb

#import needed libraries
import tensorflow.keras
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
import matplotlib.pyplot as plt

def preprocess_data():
    max_features = 10000  # top 10000 words in imdb
    maxlen = 500  # restrict words in each review to 500
    batch_size = 32

    print('Loading data...')
    (input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
    print(len(input_train), 'train sequences')
    print(len(input_test), 'test sequences')

    print('Pad sequences (samples x time)')
    input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
    input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
    print('input_train shape:', input_train.shape)
    print('input_test shape:', input_test.shape)

    return input_train, y_train, input_test, y_test

def build_model():
    #model consists of a first Embedding layers to learn good embeddings from 
    #data those weights then sent to RNN layers which provides its final output
    #to a Dense layer with sigmoid activation
    #loss function binary_crossentropy optimizer rmsprop and evaluation metric accuracy
    max_features = 10000  # top 10000 words in imdb
    model = Sequential()
    model.add(Embedding(max_features, 32))
    model.add(SimpleRNN(32))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
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
    input_train, y_train, input_test, y_test = preprocess_data()
    model = build_model()
    model.summary()
    history = model.fit(input_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
    
    #plot train vs validation loss and accuracy
    plot(history)
    print("****Test Results****")
    model.evaluate(input_test, y_test)

if __name__ == "__main__":
    main()    