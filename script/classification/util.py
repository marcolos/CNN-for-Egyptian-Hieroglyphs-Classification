import cv2
import numpy as np
import os
import shutil
import datetime
import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split  # Library for split dataset into train and test dataset

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras 
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten, MaxPooling2D
from keras.utils import plot_model
from keras.optimizers import SGD,Adam
from keras.callbacks import ModelCheckpoint,CSVLogger,LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator




# CREA UNA LISTA DI TUTTE LE IMMAGINI NELLA CARTELLA
def create_filelist(path, format):
    filelist = []
    for root, dirs, files in os.walk(path):  # os.walk returns a generator, that creates a tuple of values (current_path, directories in current_path, files in current_path).
        for file in files:
            if(file.endswith("."+format)):
                #append the file name to the list
                filelist.append(os.path.join(root,file))
    #print('PNG images found in ' + path + ': ', len(filelist))            
    return filelist


# RITORNA LA LABEL DELL'IMMAGINE DATO IL SUO PATH
def path_to_label(path):
    file_name_parts = path.split('/')
    img_name = file_name_parts[-1]
    img_name_parts = img_name.split('_')
    return img_name_parts[-1].split('.')[0]


# RITORNA UNA LISTA DI IMMAGINI DEL DATASET E UNA LISTA DI LABEL
def load_dataset(path, formato):
    filelist = create_filelist(path, formato)
    X, y = [], []
    for file in filelist:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE) #non è normalizzata tra 0 ed 1
        X.append(img)
        y.append(path_to_label(file))
    return X, y

                
        
# RITORNA LE LABELS CHE HANNO IN COMUNE I DUE DATASET
def check_labels_in_common(labels_list1, labels_list2):
    labels_comune = []
    for l in labels_list1:
        for m in labels_list2:
            if l == m:
                labels_comune.append(l)
    return labels_comune


# RITORNA IL NUMERO DELLE LABELS PER OGNI CLASSE 
def get_labels_number_in_category(labels, label_enc=None, view=False, ordina=True):
    if labels.ndim == 2:
        labels = categorical_to_decoded(labels, label_enc)
        all_labels = label_enc.classes_
    else:
        all_labels = list(set(labels))
    dict_labels = dict.fromkeys(set(all_labels), 0) 
    for l in labels:
        dict_labels[l] = dict_labels[l] + 1
    #sorted
    if ordina == True:
        sorted_dict = {}
        sorted_keys = sorted(dict_labels, key=dict_labels.get, reverse=True) 
        for w in sorted_keys:
            sorted_dict[w] = dict_labels[w]
        dict_labels = sorted_dict
    #view  
    if view == True:
        for k, v in dict_labels.items():
            print(k, v)
        print("tot labels number: " + str(len(dict_labels)))
    return dict_labels


# DETTAGLI DEL DATASET
def train_test_labels_analysis(y_train, y_test, label_enc=None):
    dict_y_train = get_labels_number_in_category(y_train, label_enc)
    dict_y_test = get_labels_number_in_category(y_test, label_enc)
    n_labels_train, n_labels_test = len(dict_y_train), len(dict_y_test)
        
    for k, v in dict_y_train.items():
        print(k, v , dict_y_test[k])
    print("labels train number: " + str(n_labels_train))
    print("labels test number: " + str(n_labels_test))

    
# CONVERTE DA CATEGORICAL A LISTA DECODATA
def categorical_to_decoded(y, label_enc):
    return label_enc.inverse_transform(np.argmax(y, axis=1))

    
# RIMUOVE DA DATASET1 LE IMMAGINI CON LABEL CON CONTENUTE IN DATASET2
def adjust_dataset1_to_dataset2(dataset_images_1, dataset_labels_1, dataset_images_2, dataset_labels_2):
    diff_labels_2 = list(set(dataset_labels_2))  # prendo tutte le label diverse di dataset2
    diff_labels_1 = list(set(dataset_labels_1))
    labels_comune = check_labels_in_common(dataset_labels_2, dataset_labels_1)
    intersect = [l for l in diff_labels_1 if l not in labels_comune]  #label rimanenti in dataset2
    dataset_images_1 = [img for i,img in enumerate(dataset_images_1) if dataset_labels_1[i] not in intersect]
    dataset_labels_1 = [label for i,label in enumerate(dataset_labels_1) if dataset_labels_1[i] not in intersect]
    return dataset_images_1, dataset_labels_1


# AGGIUNGE DIMENSIONE EXTRA A IMAGE E LABEL
def add_extra_dim(imgs, labels, n_classes):
    if imgs.ndim == 3:
        imgs = imgs.reshape((imgs.shape[0], imgs.shape[1], imgs.shape[2], 1))
    else:
        print("error: imgs dataset dimension is " +str(imgs.ndim)+" instead 3")
        return
    if labels.ndim == 1:
        labels = to_categorical(labels, num_classes=n_classes)
    else:
        print("error: labels dataset dimension is " +str(labels.ndim)+" instead 1")
        return
    return imgs, labels

# RIMUOVE DIMENSIONE EXTRA A IMAGE E LABEL
def remove_extra_dim(imgs, labels, label_enc):
    if imgs.ndim == 4:
        imgs = imgs.reshape((imgs.shape[0], imgs.shape[1], imgs.shape[2]))
    else:
        print("error: imgs dataset dimension is " +str(imgs.ndim)+" instead 3")
        return
    if labels.ndim == 2:
        labels = categorical_to_decoded(labels, label_enc=label_enc)
    else:
        print("error: labels dataset dimension is " +str(labels.ndim)+" instead 1")
        return
    return imgs, labels

        
# AUMENTO DEL DATASET
def data_augmentation(datagen, images, labels, n_aug):
    it = datagen.flow(images, labels, batch_size=1)
    for i in range(0, n_aug):
        next_it = next(it)
        image = next_it[0]
        label = next_it[1]
        images = np.append(images, image, axis= 0)
        labels = np.append(labels, label, axis=0)
        #print("augmentation of " + str(i+1) + " data")
        #plt.imshow(image, cmap="gray")
        #plt.show()
    return images, labels

         
        
def plot_train_val_test(history, history_test):
    fig, (ax1, ax2) = plt.subplots(2, figsize=(6,7))
    fig.subplots_adjust(hspace=0.5, top=0.94, bottom=0.08)
    #fig.suptitle('Resnet50', fontsize=16)

    ax1.plot(range(1, len(history['accuracy'])+1), history['accuracy'])
    ax1.plot(range(1, len(history['val_accuracy'])+1), history['val_accuracy'])
    if history_test != None:
        ax1.plot(range(1, len(history_test['test_accuracy'])+1), history_test['test_accuracy'])
    ax1.set_title('ATCNet - model accuracy')
    ax1.set(xlabel='epoch', ylabel='accuracy')
    if history_test != None:
        ax1.legend(['train', 'val', 'test'], loc='lower right')
    else:
        ax1.legend(['train', 'val'], loc='lower right')
    ax1.grid(linestyle='--')

    ax2.plot(range(1, len(history['loss'])+1), history['loss'])
    ax2.plot(range(1, len(history['val_loss'])+1), history['val_loss'])
    if history_test != None:
        ax2.plot(range(1, len(history_test['test_loss'])+1), history_test['test_loss'])
    ax2.set_title('ATCNet - model loss')
    ax2.set(xlabel='epoch', ylabel='loss')
    if history_test != None:
        ax2.legend(['train', 'val', 'test'], loc='upper right')
    else:
        ax2.legend(['train', 'val'], loc='upper right')
    ax2.grid(linestyle='--')

    plt.show()

    


def get_prediction_data(predictions, X_test, y_test, label_enc, summary=False, details=False, 
                        plot=(None,0,None), y_train=None):
    print(X_test.shape)
    # plot = (plot_type, plot_n)
    # plot_type  0 - only corrected pred
    #            1 - only wrong pred
    #            2 - corrected + wrong pred
    #            None
    # plot_n     int - number of images to plot
    #            "all" plot all images
    # only_label_to_print     "S29" - label of images to plot
    #            None
    if plot[0] is not None:
        plot_type = plot[0]
    else:
        plot_type = None
    plot_n = plot[1]
    only_label_to_print = plot[2]
    pred_corr = 0
    pred_wrong = 0
    plot_counter = 0
    y_test_decoded = label_enc.inverse_transform(np.argmax(y_test, axis=1))
    d_pred_corr = dict.fromkeys(set(y_test_decoded), 0)
    d_tot_label = dict.fromkeys(set(y_test_decoded), 0)
    if y_train is not None:
        d_train_label = get_labels_number_in_category(y_train, label_enc)
        
    for i,prediction in enumerate(predictions):
        p = np.argmax(prediction)
        true_label_enc = np.argmax(y_test[i])
        true_label = label_enc.inverse_transform([true_label_enc])[0]
        d_tot_label[true_label] = d_tot_label[true_label] +1
        
        if plot_n is "all":
            plot_n = len(predictions)
        
        if p == true_label_enc:
            pred_corr = pred_corr + 1
            d_pred_corr[true_label] = d_pred_corr[true_label] +1
            if(summary is True):
                print("test " + str(i+1)+"/"+str(len(predictions)) + " corrected prediction  " + "prediction = "+ str(p) +"  true_label_enc = "+ str(true_label_enc))
            
            if plot_type is 0 or plot_type is 2 :  
                if plot_counter < plot_n:
                    if only_label_to_print is not None:
                        if str(true_label) == only_label_to_print:
                            plt.figure(i)
                            plt.imshow(X_test[i], cmap="gray")
                            plt.title("test "+str(i+1)+"/"+str(len(predictions))+" - "+str(pred_corr)+"° corrected - " +"label "+ str(true_label))
                            plot_counter = plot_counter + 1
                    else:        
                        plt.figure(i)
                        plt.imshow(X_test[i], cmap="gray")
                        plt.title("test "+str(i+1)+"/"+str(len(predictions))+" - "+str(pred_corr)+"° corrected - " +"label "+ str(true_label))
                        plot_counter = plot_counter + 1
                    
        else:
            pred_wrong = pred_wrong + 1
            if summary is True:
                print("test " + str(i+1)+"/"+str(len(predictions)) + " wrong prediction")
            if plot_type is 1 or plot_type is 2:
                if plot_counter < plot_n:   
                    if only_label_to_print is not None:
                        if str(true_label) == only_label_to_print:
                            plt.figure(i)
                            plt.imshow(X_test[i], cmap="gray")
                            plt.title("test "+str(i+1)+"/"+str(len(predictions))+" - "+str(pred_wrong)+"° wrong prediction - " +"true_label:" +str(true_label)+ " - pred_label:" +str(label_enc.inverse_transform([p])[0]))
                            plot_counter = plot_counter + 1
                    else:
                        plt.figure(i)
                        print(X_test[i].shape)
                        plt.imshow(X_test[i], cmap="gray")
                        plt.title("test "+str(i+1)+"/"+str(len(predictions))+" - "+str(pred_wrong)+"° wrong prediction - " +"true_label:" +str(true_label)+ " - pred_label:" +str(label_enc.inverse_transform([p])[0]))
                        plot_counter = plot_counter + 1
    
    if summary is True:
        print("-------")
        print("Corrected predictions: " + str(pred_corr) + "/" + str(len(predictions)) )
        print("\n")
    
    if details is True:
        if y_train is None:
            for key in d_tot_label:
                print(str(round(d_pred_corr[key]/d_tot_label[key]*100)) +"% "+ str(d_pred_corr[key])+"/"+str(d_tot_label[key]) +" "+ key )
        else:
            for key in d_train_label:
                if key in d_tot_label:
                    print(str(d_train_label[key]) + " "+ str(round(d_pred_corr[key]/d_tot_label[key]*100)) +"% "+ str(d_pred_corr[key])+"/"+str(d_tot_label[key]) +" "+ key )    
                else:
                    print(str(d_train_label[key]) + " No images in test of label "+ str(key))
        print("-------")
        print("Correct prediction: " +str(sum(d_pred_corr.values()))+"/"+str(sum(d_tot_label.values())))
  
    if plot_type is 0 or plot_type is 1 or plot_type is 2 :
        plt.show()


def lr_schedule_tl(epoch):
    initial_lr = 0.0001
    lr = initial_lr * ((1/2)**(epoch/30))
    print('current learning rate is %2.8f' %lr)
    return lr


def lr_schedule_ft(epoch):
    initial_lr = 0.00001
    lr = initial_lr * ((1/2)**(epoch/30))
    print('current learning rate is %2.8f' %lr)
    return lr


def lr_schedule_semistatic(epoch):
    initial_lr = 0.0003
    lr = initial_lr * ((1/2)**(epoch/100) )
    print('current learning rate is %2.8f' %lr)
    return lr


def lr_schedule_exp(epoch):
    initial_lr = 0.0005
    lr = initial_lr * ((1/2)**(epoch/15) )
    print('current learning rate is %2.8f' %lr)
    return lr


def lr_schedule(epoch):
    initial_lr = 0.001
    every = 15
    lr = initial_lr*(1/2)**(math.floor(epoch/every))
    print('current learning rate is %2.8f' %lr)
    return lr



def lr_schedule_exp2(epoch):
    initial_lr = 0.001
    k = 0.05
    lrate = initial_lrate * exp(-k*epoch)
    print('current learning rate is %2.8f' %lr)
    return lrate



# SALVATAGGIO DATASET

#salva dataset su disco
def save_dataset(path, X, y, formato="jpg", subpath=False, horizontal_flip=False):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)

    if subpath is False:  # salva tutti i file in path(non crea sottocartelle)
        for i,img in enumerate(X):
            flip = cv2.flip(img, 1)
            #print("img",np.max(img), np.min(img))
            #print("flip", np.max(flip), np.min(flip))
            cv2.imwrite(path + "/" + "%04d" % (i+1) + "_" + str(y[i]) + "." + formato, img)
            if horizontal_flip is True:
                cv2.imwrite(path + "/" + "%04d" % (i+1) + "HF_" + str(y[i]) + "." + formato, cv2.flip(img, 1)) 
        print("dataset created")

    else:  # salva tutti i file in subpath(crea una cartella per ogni classe)
        subpath_list = []
        for i,img in enumerate(X):
            flip = cv2.flip(img, 1)
            print("img",np.max(img), np.min(img))
            print("flip", np.max(flip), np.min(flip))
            if y[i] not in subpath_list:
                os.mkdir(path+"/"+y[i])
                subpath_list.append(y[i])
            cv2.imwrite(path + "/" + y[i] + "/" + "%04d" % (i+1) + "_" + str(y[i]) + "." + formato, img)   
            if horizontal_flip is True:
                cv2.imwrite(path + "/" + y[i] + "/" + "%04d" % (i+1) + "HF_" + str(y[i]) + "." + formato, cv2.flip(img, 1)) 
        print("dataset created")
