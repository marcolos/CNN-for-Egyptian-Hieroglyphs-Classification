import util_img

from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import cv2


def test():
    
    # load the image
    img = load_img('/Users/marco/Desktop/tesi/campioni/campioni_processed/CampioniL_V3(200x200)/wnn - disegno/0013_E34.jpg', color_mode = "grayscale")
    # convert to numpy array
    data = img_to_array(img)
    # expand dimension to one sample
    samples = expand_dims(data, 0)
    print(samples.shape)
    # create image data augmentation generator
    datagen = ImageDataGenerator(
        rotation_range=40, # rotation
        #width_shift_range=0.15, # horizontal shift
        #height_shift_range=0.2, # vertical shift
        #zoom_range=0.5, # zoom
        horizontal_flip=False) # horizontal flip
        #brightness_range=[0.2,1.2]) # brightness

    #datagen = ImageDataGenerator(width_shift_range=[-1,1])
    
    # prepare iterator
    it = datagen.flow(samples, batch_size=1)
    # generate samples and plot
    for i in range(9):
        # define subplot
        plt.subplot(330 + 1 + i)
        # generate batch of images
        batch = it.next()
        # convert to unsigned integers for viewing
        image = batch[0]
        # plot raw pixel data
        plt.imshow(image, cmap="gray")
        cv2.imwrite("/Users/marco/Desktop/prova/img"+str(i)+".jpg", image) 
    # show the figure
    plt.show()



def aug_dataset(X_input, y_input, path_to_save, save=False, horizontal_flip=False):
    
    #calcolo il valore medio dei campioni del dataset
    dic = get_labels_number_in_category(y_input, label_enc, view=False)
    mean = int(round(np.mean([int(i) for i in dic.values()])))
    print(mean)
    
    X, y = remove_extra_dim(X_input, y_input, label_enc)

    
    #DATA AUGMENTATION
    datagen = ImageDataGenerator(
        rotation_range=2, # rotation
        width_shift_range=0.1, # horizontal shift
        height_shift_range=0.1, # vertical shift
        zoom_range=0.05, # zoom
        horizontal_flip=False, # horizontal flip
        brightness_range=[0.2,1.2]) # brightness
    #datagen.fit(X_train)

    
    dataset_images_aug, dataset_labels_aug = [],[]
    all_labels = list(set(y))
    for index_l,l in enumerate(all_labels):
        img_one_class = []
        for i,label in enumerate(y):
            if label == l:
                img_one_class.append(X[i])
        img_one_class = np.asarray(img_one_class)
        label_one_class = [l for i in range(img_one_class.shape[0])]
        label_one_class = label_enc.transform(label_one_class)
        label_one_class = np.asarray(label_one_class)
        img_one_class, label_one_class = add_extra_dim(img_one_class, label_one_class, n_classes)
        
        if img_one_class.shape[0] < mean:
            n_aug = mean - img_one_class.shape[0]
        else:
            n_aug = 0
    
        #if img_one_class.shape[0] > 10:
        #    n_aug = int(X.shape[0]/(img_one_class.shape[0] * 100))
        #else:
        #    n_aug = int(X.shape[0]/(img_one_class.shape[0] * 100))
        print(label_enc.inverse_transform(np.argmax(label_one_class, axis=1))[0], img_one_class.shape[0], n_aug)
        img_one_class, label_one_class = data_augmentation(datagen, img_one_class, label_one_class, n_aug)
        

        if index_l == 0:
            dataset_images_aug = img_one_class
            dataset_labels_aug = label_one_class
        else:
            dataset_images_aug = np.append(dataset_images_aug, img_one_class, axis= 0)
            dataset_labels_aug = np.append(dataset_labels_aug, label_one_class, axis= 0)

    if save==True:
        save_dataset(path_to_save, dataset_images_aug, categorical_to_decoded(dataset_labels_aug, label_enc), horizontal_flip=horizontal_flip)
    
    return dataset_images_aug, dataset_labels_aug


test()
#X_aug, y_aug = aug_dataset(X_test, y_test, path_to_save="/Users/marco/Desktop/new/test_aug"
#                           , save=True, horizontal_flip=True)