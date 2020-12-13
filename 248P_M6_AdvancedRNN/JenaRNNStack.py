"""
University of California, Irvine
MSWE - 248P
Sherlin Mary Koshy (smkoshy)
"""
#Reference URL :https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/6.3-advanced-usage-of-recurrent-neural-networks.ipynb
#import needed libraries
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt

def load_normalize():
    #read in file and obtain lines
    data_dir = '/home/smkoshy/248PNeuralNW/248P_Module6/jena'
    fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')
    f = open(fname)
    data = f.read()
    f.close()

    lines = data.split('\n')
    header = lines[0].split(',')
    lines = lines[1:]

    #covnvert all values to float
    float_data = np.zeros((len(lines), len(header) - 1))
    for i, line in enumerate(lines):
        values = [float(x) for x in line.split(',')[1:]]
        float_data[i, :] = values
      
    #normalize all values
    mean = float_data[:200000].mean(axis=0)
    float_data -= mean
    std = float_data[:200000].std(axis=0)
    float_data /= std  

    return float_data   

def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    #min/max indexes- range from which to draw data (train/val/test have different ranges)          
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets

def build_model(data):
    """
    Code originally includes recurrent-dropout but eliminating in my implementation as per
    author's own comment in https://github.com/keras-team/keras/issues/8935
    Using tanh activation as per : https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU#used-in-the-notebooks_1
    """
    #generate model with a Stcked Gated Recurrent Unit-similar to LSTM but more streamlined
    #since RNNs are being stacked-intermediate layers should return full sequence of outputs
    #ouput layers is a single neuron in dense layer without any activation 
    #This is because we need a linear output(regression problem)
    #rmsprop optimizer with mean absolute error as loss function
    model = Sequential()
    model.add(layers.GRU(32, dropout=0.1, return_sequences=True, input_shape=(None, data.shape[-1])))
    model.add(layers.GRU(64, dropout=0.1, activation='tanh'))
    model.add(layers.Dense(1))

    model.compile(optimizer=RMSprop(), loss='mae')

    return model

def plot(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

def main():
    data = load_normalize()

    #parameters that determine our problem
    lookback = 1440 #timesteps back data should go - 10 days 
    step = 6 #period at which we sample data, once every hour (every 6 timesteps where 1 timestep is 10 mins )
    delay = 144 #target time step in future (24 hours)
    batch_size = 128 #number of data samples per batch
    
    #generators for train/val/test data
    train_gen = generator(data, lookback=lookback, delay=delay,
                      min_index=0, max_index=200000, shuffle=True,
                      step=step, batch_size=batch_size)
    val_gen = generator(data, lookback=lookback, delay=delay,
                    min_index=200001, max_index=300000,
                    step=step, batch_size=batch_size)
    test_gen = generator(data, lookback=lookback, delay=delay,
                     min_index=300001, max_index=None,
                     step=step, batch_size=batch_size)
    
    #Compute steps needed to break out of generator when entire validation/test set is seen
    val_steps = (300000 - 200001 - lookback) // batch_size
    test_steps = (len(data) - 300001 - lookback) // batch_size                 

    #build model with Stacked RNNs
    model=build_model(data)

    #train model
    history = model.fit(train_gen, steps_per_epoch=500, epochs=40,
                              validation_data=val_gen, validation_steps=val_steps)
    plot(history)
    #evaluate on test data
    print("**********Test Results*****")
    model.evaluate(test_gen, steps=test_steps)


if __name__ == "__main__":
    main()