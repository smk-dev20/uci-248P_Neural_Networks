"""
University of California, Irvine
MSWE - 248P
Sherlin Mary Koshy (smkoshy)
"""
#Reference URL :https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/5.3-using-a-pretrained-convnet.ipynb

#import necessary libraries
import os,sys
import tensorflow.keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import numpy as np
import matplotlib.pyplot as plt

def extract_features(directory, sample_count):
    #Import VGG16 network pretrained on imagenet, not including the top-most dense layer
    #this convbase is used to extract features from our dataset
    conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    
    datagen = ImageDataGenerator(rescale=1./255)
    batch_size = 20
    
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(directory, target_size=(150, 150),
        batch_size=batch_size, class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

def build_model():
    #build model for classification 1 hidden layer 256 neurons activation-relu
    #dropout layer to reduce overfit output layer of 1 neuron with sigmoid activation for cat/dog classification
    #optimizer-rmsprop loss function-binary_crossentropy evaluation metric accuracy
    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizers.RMSprop(lr=2e-5), loss='binary_crossentropy', metrics=['acc'])
    return model
    
def plot_graphs(history):
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
    # The directory where smaller dataset is stored
    base_dir = sys.argv[1]
    print("Path to dataset: {}".format(base_dir))
    
    #specify path to dataset
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')

    #obtain features from dataset using convbase - Fast feature extraction method 
    train_features, train_labels = extract_features(train_dir, 2000)
    validation_features, validation_labels = extract_features(validation_dir, 1000)
    test_features, test_labels = extract_features(test_dir, 1000)
    
    #reshape features to make them suitable for densely connected layer
    train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
    validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
    test_features = np.reshape(test_features, (1000, 4 * 4 * 512))    
    
    #build classifier
    model=build_model()
    #fit obtained training features to classifier
    history = model.fit(train_features, train_labels, epochs=30,
                    batch_size=20, validation_data=(validation_features, validation_labels))
                    
    #save trained model  
    model.save('cats_and_dogs_pretrained_1.h5')
    #plot graphs of training vs validation accuracy and loss                
    plot_graphs(history)                      
    
    score = model.evaluate(test_features, test_labels, verbose = 0) 

    print("***Test Accuracy***") 
    print("Test accuracy:{}".format(score[1]))
                    
    
        
        
if __name__=="__main__":
    main()