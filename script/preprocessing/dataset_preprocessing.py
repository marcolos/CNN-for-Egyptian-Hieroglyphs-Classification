import os
import shutil
import cv2
import numpy as np
import statistics
from scipy import stats
import time
import matplotlib.pyplot as plt





# CHECK VECCHIA VERSIONE CON adjust_white_contorned_img E resize_img

file = "/Users/marco/Desktop/tesi/CampioniL/Aiyn - disegno/0029_D36.jpg"


img_gray = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
plot1 = plt.figure(1)
plt.imshow(img_gray, cmap="gray")
plt.colorbar()


img_gray = adjust_white_contorned_img(img_gray)
plot2 = plt.figure(2)
plt.imshow(img_gray, cmap="gray")
plt.colorbar()


img_gray = resize_img(img_gray, 50, 75, NEAR_PIXEL = 5)
plot3 = plt.figure(3)
plt.imshow(img_gray, cmap="gray")
plt.colorbar()


plt.show()


# In[ ]:


# CHECK SCONTORNO DAL BIANCO 

file = "/Users/marco/Desktop/tesi/CampioniL/Aiyn - disegno/0029_D36.jpg"

img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
plot1 = plt.figure(1)
plt.imshow(img, cmap="gray")
plt.colorbar()

thresh1 = cv2.threshold(img,250,255,cv2.THRESH_BINARY_INV)[1]/255
print(thresh1)

print(img.shape)
print(thresh1.shape)
plot2 = plt.figure(2)
plt.imshow(thresh1, cmap="gray")
plt.colorbar()


# In[ ]:


# CHECK SCONTORNO DAL BIANCO CON TUTTO IL DATASET

# dataset path
path = '/Users/marco/Desktop/tesi/CampioniL'
filelist = create_filelist(path) 

for i,file in enumerate(filelist):
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

    ret,thresh1 = cv2.threshold(img,250,255,cv2.THRESH_BINARY)

    plt.figure(i)
    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(thresh1, cmap='gray')

plt.show()


# In[ ]:


# CHECK RESIZE 

fileA = "/Users/marco/Desktop/tesi/CampioniL/Aiyn - disegno/0029_D36.jpg"
fileB = "/Users/marco/Desktop/prova/CampioniL/di - incisione/0004_X8.jpg"

img_grayA = cv2.imread(fileA, cv2.IMREAD_GRAYSCALE)
#img_grayA2 = adjust_white_contorned_img(img_grayA)
#img_grayA2 = resize_img(img_grayA2, 50, 75, NEAR_PIXEL=5)
img_grayA2 = refact_img(img_grayA, 50, 75, NEAR_PIXEL = 5)

img_grayB = cv2.imread(fileB, cv2.IMREAD_GRAYSCALE)
#img_grayB2 = adjust_white_contorned_img(img_grayB)
#img_grayB2 = resize_img(img_grayB2, 50, 75, NEAR_PIXEL=5)
img_grayB2 = refact_img(img_grayB, 50, 75, NEAR_PIXEL = 5)



plot1 = plt.figure(1)
plt.imshow(img_grayA, cmap="gray")
plt.colorbar()


plot2 = plt.figure(2)
plt.imshow(img_grayA2, cmap="gray")
plt.colorbar()

plot1 = plt.figure(3)
plt.imshow(img_grayB, cmap="gray")
plt.colorbar()


plot2 = plt.figure(4)
plt.imshow(img_grayB2, cmap="gray")
plt.colorbar()


# In[ ]:


fileA = "/Users/marco/Desktop/tesi/CampioniL/Aiyn - disegno/0029_D36.jpg"
img_grayA = cv2.imread(fileA, cv2.IMREAD_GRAYSCALE)


# # Main

# In[ ]:


# dataset path

path = '/Users/marco/Desktop/tesi/CampioniL'
#path = '/Users/marco/Desktop/tesi/Dataset_Egyptian_hieroglyphs/Manual/Preprocessed'

filelist = create_filelist(path, formato="jpg")          


#CREO DATASET 
subpath_list = []
new_dataset_path = '/Users/marco/Desktop/CampioniL_V3'
#new_dataset_path = '/Users/marco/Desktop/Oland_campioni'

if os.path.isdir(new_dataset_path):
    shutil.rmtree(new_dataset_path)

os.mkdir(new_dataset_path)
count = 0
for file in filelist:
    filename, subpath = file.split('/')[-1], file.split('/')[-2]
    if subpath not in subpath_list:
        os.mkdir(new_dataset_path + "/" + subpath)
        subpath_list.append(subpath)
    

    img_gray = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    #img_gray = adjust_white_contorned_img(img_gray)
    #img_gray = resize_img(img_gray, 50, 75, NEAR_PIXEL=5)
    
    img_gray = refact_img(img_gray, 200, 200, NEAR_PIXEL=1)
    
    cv2.imwrite(new_dataset_path + "/" + subpath + "/" + filename, img_gray)
    count = count+1
    print(str(count) +"/"+ str(len(filelist)) + " image saved")
