"""
University of California, Irvine
MSWE - 248P
Sherlin Mary Koshy (smkoshy)
"""
#Reference URL : https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/6.1-using-word-embeddings.ipynb

#import needed libraries
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
import numpy as np
import matplotlib.pyplot as plt

def load_imdb(imdb_dir):
    #read in path of imdb dataset
    train_dir = os.path.join(imdb_dir, 'train')
    test_dir = os.path.join(imdb_dir, 'test')
 
    train_labels = []
    train_texts = []
    test_labels = []
    test_texts = []
    #obtain train and test reviews and labels
    for label_type in ['neg', 'pos']:
        train_dir_name = os.path.join(train_dir, label_type)
        for fname in os.listdir(train_dir_name):
            if fname[-4:] == '.txt':
                f = open(os.path.join(train_dir_name, fname))
                train_texts.append(f.read())
                f.close()
                if label_type == 'neg':
                    train_labels.append(0)
                else:
                    train_labels.append(1)
                    
        for label_type in ['neg', 'pos']:
            test_dir_name = os.path.join(test_dir, label_type)
            for fname in os.listdir(test_dir_name):
                if fname[-4:] == '.txt':
                    f = open(os.path.join(test_dir_name, fname))
                    test_texts.append(f.read())
                    f.close()
                    if label_type == 'neg':
                        test_labels.append(0)
                    else:
                        test_labels.append(1)

    return train_texts, train_labels, test_texts, test_labels                

def tokenize_data(texts,labels):
    maxlen = 100  # limiting review length to 100 words
    training_samples = 200  # 200 reviews for training(small input size)
    validation_samples = 10000  # Validating on 10000 samples
    max_words = 10000  # Only taking top 10,000 words in the dataset

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=maxlen)

    labels = np.asarray(labels)
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    # Shuffle the data, since directory ordering would have loaded all neg and then all pos reviews
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    #Split data into training and validation sets
    x_train = data[:training_samples]
    y_train = labels[:training_samples]
    x_val = data[training_samples: training_samples + validation_samples]
    y_val = labels[training_samples: training_samples + validation_samples]

    return x_train, y_train, x_val, y_val, word_index

def preprocess_embeddings(glove_dir):
    #obtained pretrained glove embeddings
    embeddings_index = {}
    f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index

def build_embedding_matrix(embeddings_index, embedding_dim, word_index):
    #word_index has all words in imdb datset restricting to top 10000
    max_words = 10000
    #restricting the embedding matrix for imdb words to embedding_dim
    embedding_matrix = np.zeros((max_words, embedding_dim))
    #for each word in imdb, lookup the glove embedding add to matrix 
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if i < max_words:
            if embedding_vector is not None:
                # Words not found in glove embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
    return embedding_matrix   

def build_model(embedding_matrix): 
    max_words = 10000
    embedding_dim = 100
    maxlen = 100
    #first layer is embedding layer having weights learnt from glove for our review words
    #freezing this layer so weights are not modified during training
    #second hidden layer is Dense with 32 neurons and relu activation
    #output dense layer of 1 neuron for classification with sigmoid activation
    #optimizer-rmsprop loss function-binary_crossentropy and evaluation metric accuracy
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    """
    Removing lines marked 1 and 2 would make our model one that learns
    embeddings from data
    """
    model.layers[0].set_weights([embedding_matrix]) #1
    model.layers[0].trainable = False              #2
    model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              metrics=['acc'])
    return model

def plot(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

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
    imdb_path = "/home/smkoshy/248PNeuralNW/248P_Module6/aclImdb/"
    glove_path = "/home/smkoshy/248PNeuralNW/248P_Module6/glove/"
    #obtain training and test data from imdb data set
    texts, labels, test_texts, test_labels = load_imdb(imdb_path)
    
    #tokenize the training and validation data and get word index of our reviews dataset for embedding
    x_train, y_train, x_val, y_val, word_index = tokenize_data(texts,labels)
    
    #obtain global word embeddings from glove
    #glove will have a larger number of words than in imdb since it was trained on larger dataset (eg. wikipedia)
    embeddings_index = preprocess_embeddings(glove_path)
    
    #generate the embedding for imdb words by looking up the words in glove embedding index
    embedding_dim = 100 #restricting imdb review words to have 100 embdedding dimensionality
    embedding_matrix = build_embedding_matrix(embeddings_index, embedding_dim, word_index)
    
    #build Neural Network model with the embedding for imdb words as the first layer
    model = build_model(embedding_matrix)
    model.summary()  

    #train on training data
    history = model.fit(x_train, y_train, epochs=10, batch_size=32,
                    validation_data=(x_val, y_val))
    model.save_weights('pre_trained_glove_model.h5')    
    #plot train and validation loss/accuracy
    plot(history)
    #tokenize test data and evaluate on model
    tokenizer = Tokenizer(num_words=10000)
    sequences = tokenizer.texts_to_sequences(test_texts)
    x_test = pad_sequences(sequences, maxlen=100)
    y_test = np.asarray(test_labels)   
    print("****Test results****")
    model.evaluate(x_test, y_test)  

if __name__=="__main__":
    main()    