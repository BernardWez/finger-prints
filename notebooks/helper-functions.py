# Imports
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools

import os
import cv2
import random
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

import keras
from keras import layers, regularizers, optimizers
from keras.applications.resnet50 import ResNet50
from keras import layers, models, regularizers, optimizers
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model, Sequential
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from keras.preprocessing import image
from keras.callbacks import EarlyStopping, ModelCheckpoint


from hyperopt import hp, fmin, tpe, STATUS_OK, STATUS_FAIL, Trials, space_eval, rand
from keras_tqdm import TQDMNotebookCallback 

def get_confusion_matrix(y_true, y_pred, cmap='Blues', normalize=False):
    """Returns the confusion matrix for a given model"""

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Define target names
    target_names = ['female', 'male']

    # Plot confusion matrix
    plot_confusion_matrix(cm, target_names, cmap='Blues', normalize=normalize)


    print('Classification Report \n', classification_report(y_true, y_pred, labels=[0, 1], target_names=target_names))

    return cm



# https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=20)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45, fontsize=14)
        plt.yticks(tick_marks, target_names, fontsize=14)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.axhline(y=0.5, color='b', linestyle='-')
    plt.grid(b=None)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=16)
    # plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass), fontsize=16)
    plt.xlabel('Predicted label', fontsize=16)
    plt.show()
  
def evaluate_model(model, generator, directory, batch_size):
    """
    Returns predictions and true labels and prints the confusion matrix and classification report.

    Arguments
    ---------
    model:      a classifier

    generator: a keras generator (should be test_generator)

    directory: directory from which generator will pull files (should be test_dir for real evaluation)

    batch_size: batch_size used in the training process
    """

    predict_generator = generator.flow_from_directory(directory, target_size=(224,224), batch_size=batch_size, class_mode='binary', shuffle=False)

    predictions = model.predict_generator(predict_generator, steps=len(predict_generator))

    # Encodes the predictions as either 0 or 1
    y_pred = [1 if pred > 0.5 else 0 for pred in predictions]
    y_true = predict_generator.classes

    cm = get_confusion_matrix(y_true, y_pred)

    return y_pred, y_true

def plotHistory(History):
    '''Generates two plots: (1) the train accuracy vs. validation accuracy and (2) train loss vs. validation loss.'''
    sns.set(style='darkgrid', rc={'figure.figsize':(14,4)})
    data = History.history
    
    acc = data['accuracy']
    val_acc = data['val_accuracy']
    epochs = range(1, len(acc)+1)
    
    loss = data['loss']
    val_loss = data['val_loss']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,4))

    fig.suptitle('Train set vs. validation set', fontsize=16)
    
    ax1.plot(epochs, acc, 'b', label='train')
    ax1.plot(epochs, val_acc, 'r', label='val')
    ax1.set_title('Accuracy', fontsize=14)
    ax1.set_xlabel('Epoch')

    ax2.plot(epochs, loss, 'b', label='train')
    ax2.plot(epochs, val_loss, 'r', label='val')
    ax2.set_title('Loss', fontsize=14)
    ax2.set_xlabel('Epoch')
    
     # Add vertical line where accuracy is maximal
    ax1.axvline(x=val_acc.index(max(val_acc))+1, color='dimgrey', linestyle='--', label='max acc')
    ax2.axvline(x=val_loss.index(min(val_loss))+1, color='dimgrey', linestyle='--', label='min loss')
    
    ax1.legend()
    ax2.legend()

    plt.show()
    print('After {} epochs the maximum validation accuracy is {:.2%}'.format(val_acc.index(max(val_acc))+1, max(val_acc)))
    print('After {} epochs the maximum validation loss is {:.2f}'.format(val_loss.index(min(val_loss))+1, min(val_loss)))

def plot_images(generator, title='Example fingerprint images'):    
    fig, axes = plt.subplots(1, 5, figsize=(17, 5))
    
    for idx, ax in enumerate(axes):
        ax.axis('off')
        x, y = generator.next()
        index = random.randint(0, 9)
        ax.imshow(x[index])
        ax.set_title('Female' if y[index] == 0 else ' Male', size=15, pad=10)
    
    fig.suptitle(title, fontsize=20)

def buildVGG16(augment=False):
    """
    Returns pre-trained VGG16 model with imagenet weights and corresponding datagenerators
    """
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.05,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   fill_mode='constant', 
                                   cval=255)

    test_datagen = ImageDataGenerator(rescale=1./255)

    if augment:
        return base_model, train_datagen, test_datagen
    
    return base_model, test_datagen, test_datagen

def buildResNet(structure, augment=False):
    """
    Returns pre-trained ResNet model (18 or 34) with imagenet weights and corresponding datagenerator
    """
    from classification_models.keras import Classifiers

    if structure == 'ResNet-18':

      ResNet18, preprocess_input = Classifiers.get('resnet18')

      base_model = ResNet18(input_shape=(224,224,3), weights='imagenet', include_top=False)
    
    else:

      ResNet34, preprocess_input = Classifiers.get('resnet34')

      base_model = ResNet34(input_shape=(224,224,3), weights='imagenet', include_top=False)

    
    train_datagen = ImageDataGenerator(shear_range=0.2,
                                       zoom_range=0.05,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       fill_mode='constant', 
                                       cval=255,
                                       preprocessing_function=preprocess_input)

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    if augment:
        return base_model, train_datagen, test_datagen
    
    return base_model, test_datagen, test_datagen

def initialize_model(model,
                     augment,
                     freeze=True,
                     trainable='none', 
                     dense_layers=1, 
                     dense_hidden_size=64,
                     dropout_rate=0, 
                     l1_reg=0, 
                     l2_reg=0, 
                     optimizer_function='SGD', 
                     learning_rate=0.001,
                     momentum=0,
                     nesterov=False,
                     decay=0,
                     print_model=False,
                     **arg):
    """
    Builds CNN using pre-trained models with option to tune layers/regularization/optimizers and returns configured model. 
    """
    
    if model == 'ResNet-18':
      pre_trained_model, train_datagen, test_datagen = buildResNet(model, augment)
    
    if model == 'ResNet-34':
      pre_trained_model, train_datagen, test_datagen = buildResNet(model, augment)

    
    if model == 'VGG16':
      pre_trained_model, train_datagen, test_datagen = buildVGG16(augment)
    
    if freeze:
      #Freeze all layers in the imported CNN
      for layer in pre_trained_model.layers:
          layer.trainable = False
    
    # Unfreeze last conv layer from the imported CNN if we aim to train the last layer
    if trainable=='last_layer':
        pre_trained_model.layers[-2].trainable = True

    average_pool = GlobalAveragePooling2D()(pre_trained_model.output)
    
    add_layer = average_pool
    
    for layer in range(dense_layers):
        add_layer = Dense(dense_hidden_size, activation='relu', 
                          kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg))(add_layer)
        
        if dropout_rate > 0:
            add_layer = Dropout(rate=dropout_rate)(add_layer)
        
    # Add output layer for classificaiton
    output_layer = Dense(1, activation='sigmoid')(add_layer)

    # Build new model structure
    model_final = Model(inputs=pre_trained_model.input, outputs=output_layer)
    
    if optimizer_function == 'adam':
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        
    if optimizer_function == 'RMSprop':
        optimizer = optimizers.RMSprop(learning_rate=learning_rate)

    if optimizer_function == 'SGD':
        optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=momentum, decay=decay, nesterov=nesterov)

        
    # Configure model for training
    model_final.compile(loss='binary_crossentropy',
                        optimizer=optimizer,
                        metrics=['accuracy'])
    
    if print_model:
        model_final.summary()
    
    return model_final, train_datagen, test_datagen

def fit_model(name, train_gen, test_datagen, epochs, batch_size):
    """
    Retuns History object after training model
    """
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10)
    progress = TQDMNotebookCallback(leave_inner=False)
    filepath = name + '-weights.best.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    
    #
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=(224,224),
                                                        batch_size=batch_size,
                                                        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(val_dir,
                                                            target_size=(224,224),
                                                            batch_size=batch_size,
                                                            class_mode='binary')


    validation_generator


    # Store the keras callbacks
    callbacks = [progress, early_stopping, checkpoint]

    hist = model.fit_generator(train_generator,
                              steps_per_epoch= (1400 // batch_size),
                              callbacks=callbacks,
                              validation_data=validation_generator, 
                              validation_steps= (600 // batch_size),
                              epochs=epochs,
                              verbose=0)
    return hist
    