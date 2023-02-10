import util_img as util
import os
import shutil
import cv2
import numpy as np
import statistics
from scipy import stats
import time
import matplotlib.pyplot as plt
         

#FLIPPARE DATASET
# name = "test"
# path = '/Users/marco/Desktop/tesi/campioni/datasets(100x100)/'+ name 

# filelist = util.create_filelist(path, formato="jpg")          

# new_dataset_path = '/Users/marco/Desktop/tesi/campioni/datasets(100x100)/flipped'+ name

# if os.path.isdir(new_dataset_path):
#     shutil.rmtree(new_dataset_path)

# os.mkdir(new_dataset_path)
# count = 0
# for file in filelist:
#     filename, subpath = file.split('/')[-1], file.split('/')[-2]

#     img_gray = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    
    
#     cv2.imwrite(new_dataset_path  + "/" + filename, cv2.flip(img_gray, 1))
#     count = count+1
#     print(str(count) +"/"+ str(len(filelist)) + " image saved")



# ALTRI CHE NON SIA KFOLD
#name = "train_aug_flip"
name = "val"
#name = "test"
resize = 299
path = '/Users/marco/Desktop/tesi/campioni/datasets(100x100)/'+ name 

filelist = util.create_filelist(path, formato="jpg")          

new_dataset_path = '/Users/marco/Desktop/tesi/campioni/datasets(299x299)/'+ name + str(resize)

if os.path.isdir(new_dataset_path):
    shutil.rmtree(new_dataset_path)

os.mkdir(new_dataset_path)
count = 0
for file in filelist:
    filename, subpath = file.split('/')[-1], file.split('/')[-2]

    img_gray = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    
    img_gray = cv2.resize(img_gray, (resize,resize))
    
    cv2.imwrite(new_dataset_path  + "/" + filename, img_gray)
    count = count+1
    print(str(count) +"/"+ str(len(filelist)) + " image saved")




# K-FOLD
# name = "5"
# path = '/Users/marco/Desktop/tesi/campioni/datasets(200x200)/K-fold/'+ name

# filelist = util.create_filelist(path, formato="jpg")          

# subpath_list = []
# new_dataset_path = '/Users/marco/Desktop/tesi/campioni/datasets(100x100)/K-fold/'+name

# if os.path.isdir(new_dataset_path):
#     shutil.rmtree(new_dataset_path)

# os.mkdir(new_dataset_path)
# count = 0
# for file in filelist:
#     filename, subpath = file.split('/')[-1], file.split('/')[-2]

#     if subpath not in subpath_list:
#         os.mkdir(new_dataset_path + "/" + subpath)
#         subpath_list.append(subpath)

#     img_gray = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    
#     img_gray = cv2.resize(img_gray, (100,100))
    
#     cv2.imwrite(new_dataset_path  +"/"+ subpath +"/" + filename, img_gray)
#     count = count+1
#     print(str(count) +"/"+ str(len(filelist)) + " image saved")