# -*- coding: utf-8 -*-
"""
Created on Thu May 17 17:15:54 2018

@author: mednche

This script reads the names of the images in both Test and Train folders (which need to be unzipped),
and check whether there is any overlap between both. 

"""
import os

# Get all files in Test folder
folder = "C:/Users/mednche/Desktop/ImageRec/Test"
onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]


# Get all files in Train folder
folder = "C:/Users/mednche/Desktop/ImageRec/Train"
onlyfiles2 = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

# check that there is no overlap between the two datasets
def common_elements(list1, list2):
    return [element for element in list1 if element in list2]

common_elements(onlyfiles, onlyfiles2)

# Good, now we can combine them (100,065 images in total)
# Manually copy all images into a new folder called Merged.



