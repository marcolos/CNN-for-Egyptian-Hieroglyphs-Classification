# -*- coding: utf-8 -*-
"""util_img.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Cl5-FM-aMzHadKH-vWIqkIE9P7tcOxDF
"""

import cv2
import numpy as np
import statistics
from scipy import stats
import matplotlib.pyplot as plt
import os

# CREA UNA LISTA DI TUTTE LE IMMAGINI NELLA CARTELLA
def create_filelist(path,formato="jpg"):
    filelist = []
    for root, dirs, files in os.walk(path):  # os.walk returns a generator, that creates a tuple of values (current_path, directories in current_path, files in current_path).
        for file in files:
            if(file.endswith("."+formato)):
                #append the file name to the list
                filelist.append(os.path.join(root,file))
    #print('PNG images found in ' + path + ': ', len(filelist))            
    return filelist


# RITORNA MEDIA, STD E MODA DEL BACKGROUND
def color_backgroung(gray_img, NEAR_PIXEL = 5):
    h, w = int(gray_img.shape[0]) , int(gray_img.shape[1])
    background = []
    for row in gray_img:
        background = np.concatenate((background , row[0:NEAR_PIXEL] , row[w-NEAR_PIXEL-1:w-1]), axis = None)
    avg_background, std_background = np.mean(background) , np.std(background)
    mode_background = stats.mode(background)[0]
    return avg_background, std_background, mode_background
    

# RESIZE DELL'IMMAGINE
def resize_img(gray_img, out_w, out_h, NEAR_PIXEL = 5):
    h, w = int(gray_img.shape[0]) , int(gray_img.shape[1])
    ratio = w/h
    out_ratio = out_w/out_h
    avg_background, std_background, mode_background = color_backgroung(gray_img, NEAR_PIXEL)
    
    if ratio > out_ratio:
        new_h = int(w/out_ratio)
        offset = int( (new_h - h)/2 )
        background = np.random.normal(mode_background, 15, (new_h,w)).astype('float32')
        
        #FILTRI
        #background = cv2.GaussianBlur(background, (11, 11), 0)
        background = cv2.medianBlur(background,3,0)
        #background = cv2.blur(background,(5,5))
        
        background[offset:offset+gray_img.shape[0], 0:gray_img.shape[1]] = gray_img

    elif ratio < out_ratio:
        new_w = int(h*out_ratio)
        offset = int( (new_w - w)/2 )
        background = np.random.normal(mode_background, 15, (h,new_w)).astype('float32')

        #FILTRI
        #background = cv2.GaussianBlur(background, (11, 11), 0)
        background = cv2.medianBlur(background,3,0)
        #background = cv2.blur(background,(5,5))
        
        background[0:gray_img.shape[0], offset:offset+gray_img.shape[1]] = gray_img
        
    else:
        return cv2.resize(gray_img, (out_w,out_h))
    
    resized_img = cv2.resize(background, (out_w,out_h))
    return resized_img




# AGGIUSTO LE IMMAGINI CONTORNATE DI BIANCO
def adjust_white_contorned_img(img_gray, THRESHOLD_WHITE = 240, NEAR_PIXEL = 5):
    h, w = int(img_gray.shape[0]) , int(img_gray.shape[1])
    background = []
    white_pixel_position = []
    count = 0
    
    # lato di sinistra
    for i,row in enumerate(img_gray): 
        for j,pixel in enumerate(row):  
            if pixel<THRESHOLD_WHITE:
                count = count + 1
                background.append(pixel)
                if count == NEAR_PIXEL:
                    count = 0
                    break  #esce fuori dal ciclo j
            else:
                white_pixel_position.append([i,j])

    # lato di destra            
    for i in range(0,len(img_gray)-1): 
        for j in range(len(img_gray[i])-1,0,-1):  
            pixel = img_gray[i][j]
            if pixel<THRESHOLD_WHITE:
                count = count + 1
                background.append(pixel)
                if count == NEAR_PIXEL:
                    count = 0
                    break  #esce fuori dal ciclo j
            else:
                white_pixel_position.append([i,j])
    
    if(len(white_pixel_position)> h*NEAR_PIXEL):  #entra solo se la metà dei pixel sono riconosciuti come bianchi
        background = np.array(background)
        mean_background, std_background = background.mean(axis=0), background.std(axis=0)
        mode_background = mode_background = stats.mode(background)[0]
        
        #creo il background
        background = np.random.normal(mode_background, std_background, (h,w)).astype('float32')
        background = cv2.medianBlur(background,5,0)
        
        bw = np.zeros((h, w))   #creo una matrice bw di pixel neri 
        for i,j in white_pixel_position:
            bw[i][j] = 255   #metto a bianco(255) i pixel che contengono il colore bianco nell'immagine originale

        for i,row in enumerate(bw):
            for j,pixel in enumerate(row):
                if pixel == 0:  #se 
                    background[i][j] = img_gray[i][j] 
        return background
    else:
        return img_gray

    
    
    
def make_mean_std_mode_ok(gray_img, thresh, NEAR_PIXEL = 5):
    background = []
    for i,row in enumerate(thresh):
        mul = np.multiply(row,gray_img[i]) 
        mul = mul[mul != 0]
        background = np.concatenate((background , mul[0:NEAR_PIXEL] , mul[len(mul)-NEAR_PIXEL:len(mul)]), axis = None)
    avg_background, std_background = np.mean(background) , np.std(background)
    mode_background = stats.mode(background)[0]
    return avg_background, std_background, mode_background
  

def make_mean_std_mode(gray_img, thresh, NEAR_PIXEL = 1):
    background = []
    save_first_row = True
    save_last_row = True
    for i,row in enumerate(thresh):
        mul = np.multiply(row,gray_img[i]) 
        mul = mul[mul != 0]
        
        # per salvare la prima e l'ultima riga dell'immagine di colore non bianco
        if(save_first_row is True and len(mul) > 0):
            background = np.concatenate( (background , mul), axis = None)
            save_first_row = False
            #print("salvo prima riga: ",mul)
        if(save_first_row is False and save_last_row is True): # ci entra sempre dopo che ha salvato la prima riga
            #print(len(mul),i,len(thresh)-1)
            if(len(mul) is 0):
                background = np.concatenate( (background , old_mul), axis = None)
                save_last_row = False
                #print("salvo ultima riga(1): ",old_mul) 
            
            if(i is (len(thresh)-1) and len(mul)>0):
                background = np.concatenate( (background , mul), axis = None)
                #print("salvo ultima riga(2): ",old_mul) 
                save_last_row = False
                
        background = np.concatenate((background , mul[0:NEAR_PIXEL] , mul[len(mul)-NEAR_PIXEL:len(mul)]), axis = None)
        old_mul = mul
    avg_background, std_background = np.mean(background) , np.std(background)
    mode_background = stats.mode(background)[0]
    return avg_background, std_background, mode_background



    
# RESIZE DELL'IMMAGINE
def refact_img(gray_img, out_w, out_h, NEAR_PIXEL = 1):
    h, w = int(gray_img.shape[0]) , int(gray_img.shape[1])
    ratio = w/h
    out_ratio = out_w/out_h
    
    # costruisco una matrice thresh che ha i pixel neri(=0) dove c'è il bianco nell'immagine in input
    thresh = cv2.threshold(gray_img,240,255,cv2.THRESH_BINARY_INV)[1]/255
    
    # calcolo media, std, moda
    avg_background, std_background, mode_background = make_mean_std_mode(gray_img, thresh, NEAR_PIXEL)
    
    if ratio > out_ratio: 
        new_h = int(w/out_ratio)
        offset = int( (new_h - h)/2 )
        background = np.random.normal(mode_background, 15, (new_h,w)).astype('float32')
        
        #FILTRI
        #background = cv2.GaussianBlur(background, (11, 11), 0)
        background = cv2.medianBlur(background,3,0)
        #background = cv2.blur(background,(5,5))
        
        for i,row in enumerate(thresh):
            for j,pixel in enumerate(row):
                if pixel == 1:
                    background[offset+i, j] = gray_img[i,j]

    elif ratio < out_ratio:
        new_w = int(h*out_ratio)
        offset = int( (new_w - w)/2 )
        #background = np.random.normal(mode_background, 15, (h,new_w)).astype('float32')
        background = np.random.normal(avg_background, std_background, (h,new_w)).astype('float32')
    
        #FILTRI
        #background = cv2.GaussianBlur(background, (11, 11), 0)
        background = cv2.medianBlur(background,3,0)
        #background = cv2.blur(background,(5,5))
        
        for i,row in enumerate(thresh):
            for j,pixel in enumerate(row):
                if pixel == 1:
                    background[i, offset+j] = gray_img[i,j]
        
    else:
        return cv2.resize(gray_img, (out_w,out_h))
    refacted_img = cv2.resize(background, (out_w,out_h))
    return refacted_img