import os
import cv2
import numpy as np
import random
from sklearn.model_selection import train_test_split

def getLabel(img_name):
    '''Takes image name as string and returns the associated labels'''
    # Remove the extension from image filename
    img_name = os.path.splitext(img_name)[0]
        
    # Create the image labels
    label_list = img_name.split('_')
    
    # Subject id
    subject_id = label_list[0]
    
    # Gender: code male as 1 and female as 0
    gender = 1 if label_list[2] == 'M' else 0
    
    # Hand: code left hand as 1 and right hand as 0
    hand = 1 if label_list[3] == 'Left' else 0
    
    # Finger: code fingers according to dictionary mapping below
    fingers = {'index': 0, 'little': 1, 'middle': 2, 'ring':3, 'thumb': 4}
    finger = fingers[label_list[4]] 
    
    return gender, hand, finger

def imageImport(img_dir):
    '''Imports images from a given directory'''
    
    data = []
    
    for img in os.listdir(img_dir):
        # Get the entire image path
        img_path = os.path.join(img_dir, img)
        
        # Get the image labels
        img_labels = getLabel(img)
        
        # Read the image from the image path
        img_array = cv2.imread(img_path, 0)
        
        img_resize = cv2.resize(img_array, (244, 244)) / 255
        
        stacked_img = np.stack((img_resize,)*3, axis=-1)
        
        # Append image array and image label
        data.append((stacked_img, img_labels))
        
    return np.array(data)

def genderTrainTestSplit(all_data):
    '''
    Returns X and y for gender data using undersampling
    '''
    
    data = [(datapoint[0], datapoint[1][0]) for datapoint in all_data]
    random.shuffle(data)
    
    female_data = []
    male_data = []
    
    for i in range(len(data)):
        if data[i][1] == 1:
            male_data.append(data[i])
        else:
            female_data.append(data[i])
    
    
    balanced = female_data + male_data[:len(female_data)]
    random.shuffle(balanced)
    
    x, y = zip(*balanced)
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.186991869918, shuffle=True, stratify=y)
    
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)