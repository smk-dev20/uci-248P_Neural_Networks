"""
University of California, Irvine
MSWE - 248P
Sherlin Mary Koshy (smkoshy)
"""
#Reference URL :https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/5.2-using-convnets-with-small-datasets.ipynb

#import needed libraries
import sys
import os, shutil
from os import path
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

def create_data_directories(original_dataset_dir, base_dir):
    #create directories and copy data only if not already present
    if(not(path.exists(base_dir))):
        os.mkdir(base_dir)

        # Directories for our training, validation and test splits
        train_dir = os.path.join(base_dir, 'train')
        os.mkdir(train_dir)
        validation_dir = os.path.join(base_dir, 'validation')
        os.mkdir(validation_dir)
        test_dir = os.path.join(base_dir, 'test')
        os.mkdir(test_dir)

        # Directory with our training cat pictures
        train_cats_dir = os.path.join(train_dir, 'cats')
        os.mkdir(train_cats_dir)

        # Directory with our training dog pictures
        train_dogs_dir = os.path.join(train_dir, 'dogs')
        os.mkdir(train_dogs_dir)

        # Directory with our validation cat pictures
        validation_cats_dir = os.path.join(validation_dir, 'cats')
        os.mkdir(validation_cats_dir)

        # Directory with our validation dog pictures
        validation_dogs_dir = os.path.join(validation_dir, 'dogs')
        os.mkdir(validation_dogs_dir)

        # Directory with our test cat pictures
        test_cats_dir = os.path.join(test_dir, 'cats')
        os.mkdir(test_cats_dir)

        # Directory with our test dog pictures
        test_dogs_dir = os.path.join(test_dir, 'dogs')
        os.mkdir(test_dogs_dir)

        # Copy first 1000 cat images to train_cats_dir
        fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(train_cats_dir, fname)
            shutil.copyfile(src, dst)

        # Copy next 500 cat images to validation_cats_dir
        fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(validation_cats_dir, fname)
            shutil.copyfile(src, dst)
    
        # Copy next 500 cat images to test_cats_dir
        fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(test_cats_dir, fname)
            shutil.copyfile(src, dst)
    
        # Copy first 1000 dog images to train_dogs_dir
        fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(train_dogs_dir, fname)
            shutil.copyfile(src, dst)
    
        # Copy next 500 dog images to validation_dogs_dir
        fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(validation_dogs_dir, fname)
            shutil.copyfile(src, dst)

        # Copy next 500 dog images to test_dogs_dir
        fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(test_dogs_dir, fname)
            shutil.copyfile(src, dst)

        #print count of directory contents to verify    
        print('total training cat images:', len(os.listdir(train_cats_dir))) 
        print('total training dog images:', len(os.listdir(train_dogs_dir)))
        print('total validation cat images:', len(os.listdir(validation_cats_dir)))
        print('total validation dog images:', len(os.listdir(validation_dogs_dir)))
        print('total test cat images:', len(os.listdir(test_cats_dir)))
        print('total test dog images:', len(os.listdir(test_dogs_dir)))

def build_model():
    #build CNN 4 Conv2D layers with channels 32,64,128 and 128 all with sliding window(3,3)
    # 4 MaxPool2D layers with window(2,2), Flatten before sending as input to dense layer
    # Dropout layer added to reduce overfitting
    # Dense layer of 512 neuron all hidden layers use relu activation
    # output layer of 1 neuron for binary classification hence sigmoid activation
    # optimizer-rmsprop loss function- binary_crossentropy and evaluation metric - accuracy 
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])         
    return model

def preprocess_data(base_dir):
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')
    # All images will be rescaled by 1./255 and training data is augmented to increase
    #number of input samples
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2,
    height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True,)
    #validation and test data should not be augmented
    validation_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)


    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150),
        batch_size=20, class_mode='binary')

    validation_generator = validation_datagen.flow_from_directory(validation_dir,target_size=(150, 150),
        batch_size=20, class_mode='binary')

    test_generator = test_datagen.flow_from_directory(test_dir,target_size=(150, 150),
        batch_size=20, class_mode='binary')    
     
    return train_generator, validation_generator, test_generator

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
    # The path to the directory where the original dataset was uncompressed in ICS
    original_dataset_dir = sys.argv[1]
    #print("input org path {}".format(original_dataset_dir))
    # The directory where smaller dataset will be stored
    base_dir = sys.argv[2]
    #print("input base path {}".format(base_dir))

    #create directories with cat/dog data
    create_data_directories(original_dataset_dir, base_dir)

    #preprocess data and perform data augmentation on training data
    train_generator, validation_generator, test_generator = preprocess_data(base_dir)
    print("****Verify Training data shape****")
    for data_batch, labels_batch in train_generator:
        print('data batch shape:', data_batch.shape)
        print('labels batch shape:', labels_batch.shape)
        break

    #build the CNN model
    
    model = build_model()
    print(model.summary()) 

    #train model using generator and verify against validation generator
    history = model.fit(train_generator, steps_per_epoch=100, epochs=100,
      validation_data=validation_generator, validation_steps=50)
    #save trained model  
    model.save('cats_and_dogs_small_1.h5')
    

    #plot graphs showing training vs validation loss and accuracy
    plot_graphs(history)

    test_loss, test_acc = model.evaluate(test_generator, steps=50)
    print("****Test Results***")    
    print("Test Accuracy: {}".format(test_acc))


if __name__=="__main__":
    main()

