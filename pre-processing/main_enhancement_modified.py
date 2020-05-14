# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 11:42:58 2016

@author: utkarsh
"""

import numpy as np
import cv2
import sys
import os
import time

from image_enhance import image_enhance

import concurrent.futures

def get_enhanced_img(filename):
    # Set the path to the image
    img_path = os.path.join(img_dir, filename)

    # Load the image
    img = cv2.imread(img_path)

    if(len(img.shape) > 2):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rows, cols = np.shape(img)
    aspect_ratio = np.double(rows) / np.double(cols)

    new_rows = 350            # randomly selected number
    new_cols = new_rows / aspect_ratio

    img = cv2.resize(img, (np.int(new_cols), np.int(new_rows)))

    enhanced_img = image_enhance(img)

    enhanced_img_path = os.path.join(enhanced_dir_path, filename)
    cv2.imwrite(enhanced_img_path, (255 * enhanced_img))

if __name__ == '__main__':

    t1 = time.perf_counter()

    img_dir = '/mnt/c/Users/Meekmar/Github/finger-prints/data/train/female'

    # Get list of all image names
    img_names = os.listdir(img_dir)

    # Set base path
    base_path = os.path.join(img_dir, '..')

    # Set name for enhanced directory
    enhanced_dir_path = os.path.join(base_path, img_dir.split('/')[-1] + '_enhanced')

    # Create directory for enhanced images
    os.mkdir(enhanced_dir_path)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(get_enhanced_img, img_names)

    t2 = time.perf_counter()

    print(f'Finished in {t2-t1} seconds')