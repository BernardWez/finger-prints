#coding=utf-8

try:
    import os
except:
    pass

try:
    import cv2
except:
    pass

try:
    import random
except:
    pass

try:
    import numpy as np
except:
    pass

try:
    import matplotlib.pyplot as plt
except:
    pass

try:
    import seaborn as sns
except:
    pass

try:
    from sklearn.model_selection import train_test_split
except:
    pass

try:
    import keras
except:
    pass

try:
    from keras import layers, regularizers, optimizers
except:
    pass

try:
    from keras.callbacks import EarlyStopping
except:
    pass

try:
    from hyperopt import Trials, STATUS_OK, tpe
except:
    pass

try:
    from hyperas import optim
except:
    pass

try:
    from hyperas.distributions import choice, uniform
except:
    pass
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


REAL = '/mnt/c/Users/Meekmar/Github/finger-prints/SOCOFing/Real'

# Load the data | (img, (gender-label, hand-label, finger-lable))
data = imageImport(REAL)

# Splits the train-test in the same way as the paper did (I added an additional train-validation split for tuning of the model)
X_gender_train, X_gender_val, X_gender_test, y_gender_train, y_gender_val, y_gender_test = genderTrainTestSplit(data, 1000, 230)



def keras_fmin_fnct(space):

    model = keras.Sequential()

    model.add(layers.Conv2D(space['Conv2D'], kernel_size=(3,3), input_shape=(103,96,1), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Conv2D(space['Conv2D_1'], kernel_size=(3,3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=8, activation='relu', kernel_regularizer=regularizers.l2(space['l2'])))
    model.add(layers.Dense(units=1, activation='sigmoid'))
    
    model.compile(optimizer=space['optimizer'], loss='binary_crossentropy', metrics=['accuracy'])
    
    history = model.fit(X_gender_train, y_gender_train, 
                        batch_size=space['batch_size'], 
                        epochs=space['epochs'], 
                        verbose=0, 
                        validation_data=(X_gender_val, y_gender_val))
    
    #get the highest validation accuracy of the training epochs
    validation_acc = np.amax(result.history['val_acc']) 
    print('Best validation acc of epoch:', validation_acc)
    
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}

def get_space():
    return {
        'Conv2D': hp.choice('Conv2D', [16,24,32,40]),
        'Conv2D_1': hp.choice('Conv2D_1', [16,24,32,40]),
        'l2': hp.choice('l2', [0.01, 0.05, 0.1, 0.2]),
        'optimizer': hp.choice('optimizer', ['rmsprop', 'adam', 'sgd']),
        'batch_size': hp.choice('batch_size', [32,64,128]),
        'epochs': hp.choice('epochs', [30,50,100]),
    }
