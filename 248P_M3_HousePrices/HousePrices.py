"""
University of California, Irvine
MSWE - 248P
Sherlin Mary Koshy (smkoshy)
"""
#Reference URL : https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/3.7-predicting-house-prices.ipynb

#import neeed libraries
import tensorflow.keras
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras import models
from tensorflow.keras import layers

import numpy as np
import matplotlib.pyplot as plt

#load training/test data and targets - Regression problem hence output is a continuous value
(train_data, train_targets), (test_data, test_targets) =  boston_housing.load_data()
#print(train_data.shape)
#print(test_data.shape)
#print(train_targets)

#each of the 13 features has different ranges - they need to be normailzed to take smaller
#values centered around 0
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std


#build initial small network 2 hidden layers 64 neurons each - hidden layer activation-relu
#regression problem therefore output is predicted linear house price (no activation)
#loss function - mean squared error evaluation metric- mean absolute error
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='tanh',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='tanh'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

#input sample size is very small hence use k-fold cross validation - different
#sections of input data are used as training and validation sets and then interchanged   

#Splitting data into 4 folds 1 fold is validation set and remaiing 3 are training
k = 4
num_val_samples = len(train_data) // k

#from above predictions have an error of +/-$2400 on average
#attempt to improve by training for longer and save per epoch validation score log

#memory clean-up
#from tensorflow.keras import backend as K
#K.clear_session()

"""
num_epochs = 200 #500 epochs runtime was long
all_mae_histories = []
for i in range(k):
    print('processing fold #', i)
    # Prepare the validation data: data from partition # k
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    # Prepare the training data: data from all other partitions
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)
    # Build the Keras model 
    model = build_model()
    # Train the model 500 epochs per train-validation fold, save mea 
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)

#determine average per-epoch MAE score for all folds    
average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

# omit first 10 points and replace each point with exponential moving average to see a 
# smooth curve    
def smooth_curve(points, factor=0.9):
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]
      smoothed_points.append(previous * factor + point * (1 - factor))
    else:
      smoothed_points.append(point)
  return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()    
"""

#on the above tests MAE stops improving after 80 epochs, retrain for 80 on full data (initial network)
#"""
model = build_model()
# Train it on the entirety of the data.
model.fit(train_data, train_targets,
          epochs=80, batch_size=16, verbose=0) #epoch 80 for best config 2 layers 64 neurons tanh
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print("***Test Score****")
print(test_mae_score)
#"""