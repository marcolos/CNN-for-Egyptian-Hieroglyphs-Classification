import cv2
import numpy as np
import statistics
from scipy import stats
import matplotlib.pyplot as plt
import os



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
        
        true_background = background.copy()
        plt.figure(3)
        plt.imshow(background, cmap="gray", vmin=0, vmax=255)
        plt.colorbar()

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
        true_background = background.copy()
        
        for i,row in enumerate(thresh):
            for j,pixel in enumerate(row):
                if pixel == 1:
                    background[i, offset+j] = gray_img[i,j]
        
    else:
        return true_background, cv2.resize(gray_img, (out_w,out_h))
    #refacted_img = cv2.resize(background, (out_w,out_h))
    refacted_img = background
    return cv2.resize(true_background, (out_w,out_h)) , refacted_img




file = "/Users/marco/Desktop/0013_I10.jpg"
img_rgb = cv2.imread(file, cv2.COLOR_BGR2RGB)
img_gray = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

background, refacted_img = refact_img(img_gray, 200,200)
cv2.imwrite("/Users/marco/Desktop/back2.jpg", background)
cv2.imwrite("/Users/marco/Desktop/gray2.jpg", img_gray)
# plt.figure(1)
# plt.imshow(img_rgb)
# plt.colorbar()


# plt.figure(2)
# plt.imshow(img_gray, cmap="gray")
# plt.colorbar()



# plt.figure(4)
# plt.imshow(refacted_img, cmap="gray")
# plt.colorbar()

plt.show()

