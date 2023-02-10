import tensorflow as tf
from tensorflow import keras 
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, MaxPool2D , Flatten, MaxPooling2D, BatchNormalization, Activation, Add, ReLU, SeparableConv2D, GlobalAvgPool2D, Dropout, add, GlobalAveragePooling2D
from keras.utils import plot_model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from numpy import expand_dims

from util import *
from model import *


# DATASET PREPARATION

path_train = "/Users/marco/Desktop/glyphNet/datasets/train_aug_flip"
path_val = "/Users/marco/Desktop/glyphNet/datasets/val"
path_test = "/Users/marco/Desktop/glyphNet/datasets/test"
log_filepath = "/Users/marco/Desktop/glyphNet/logs"
checkpoint_filepath = "/Users/marco/Desktop/glyphNet/weights"
weights_path = "/Users/marco/Desktop/glyphNet/weights"
     
# loading datasets from disco
X_train, y_train = load_dataset(path_train, "jpg")
X_val, y_val = load_dataset(path_val, "jpg")
X_test, y_test = load_dataset(path_test, "jpg")


# encoding  # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
label_enc = preprocessing.LabelEncoder()
label_enc.fit(y_train) 
y_train = label_enc.transform(y_train) # trasforma le label da caratteri a numeri
y_val = label_enc.transform(y_val) # trasforma le label da caratteri a numeri
y_test = label_enc.transform(y_test) # trasforma le label da caratteri a numeri

n_classes = len(list(label_enc.classes_))

# list to np array
X_train = np.asarray(X_train)
X_val = np.asarray(X_val)
X_test = np.asarray(X_test)

# add extra-dimension 
X_train, y_train = add_extra_dim(X_train, y_train, n_classes)
X_val, y_val = add_extra_dim(X_val, y_val, n_classes)
X_test, y_test = add_extra_dim(X_test, y_test, n_classes)

print('Train: ',X_train.shape, y_train.shape)
print('Val: ',X_val.shape, y_val.shape)
print('Test: ',X_test.shape, y_test.shape)



# BUILD MODEL
model = ATCNet(shape=(np.shape(X_train)[1], np.shape(X_train)[2], 1), n_classes=n_classes)

top3 = tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3")
top5 = tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5")
model.compile(optimizer=Adam(), 
            loss='categorical_crossentropy', 
            metrics=['accuracy', top3, top5])

#model.compile(optimizer=SGD(lr=0.01, momentum=0.9), 
#                  loss='categorical_crossentropy', 
#                  metrics=['accuracy'])


# TRAINING 
# log_dir = log_filepath + "/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# checkpoint = ModelCheckpoint(checkpoint_filepath + "/weights.{epoch:02d}-{val_accuracy:.2f}.hdf5" ,
#                              save_weights_only=True,
#                              monitor='val_accuracy',
#                              mode='max',
#                              save_best_only=False,
#                              verbose=1,
#                              save_freq='epoch')

# lr_scheduler = LearningRateScheduler(lr_schedule)
# history = model.fit(X_train, y_train,
#                     validation_data=(X_val, y_val), 
#                     epochs=5, batch_size=32, shuffle=True, callbacks=[checkpoint, lr_scheduler])
#np.save('/content/drive/MyDrive/ColabNotebooks/history_train.npy',history.history)


# CHOOSE THE BEST WEIGHTS to TEST
# hist_test = {}
# accuracy_max = 0
# weights_max = ""
# test_loss, test_accuracy, test_top3, test_top5 = [],[],[],[]
# for weights in os.listdir(weights_path):
#     if(weights.endswith(".hdf5")):
#         print(weights)
#         model.load_weights(weights_path +"/"+ weights)
#         l, a, top3, top5 = model.evaluate(X_test, y_test, verbose=1)
#         test_loss.append(l)
#         test_accuracy.append(a)
#         test_top3.append(top3)
#         test_top5.append(top5)
#         if accuracy_max < a:
#             accuracy_max = a
#             weights_max = weights
#         print("---------------------------------------------------------------------------------------------")
# print(weights_max + "  accuracy: " + str(accuracy_max))
# hist_test = {"test_loss":test_loss, "test_accuracy":test_accuracy, "test_top3":test_top3, "test_top5":test_top5}
#np.save('/content/drive/MyDrive/ColabNotebooks/history_test.npy',hist_test)


# model.load_weights(weights_path +"/"+weights_max)
# y_pred = model.predict(X_test)
# y_test_decoded = np.argmax(y_test, axis=1)
# y_pred_decoded = np.argmax(y_pred, axis=1)
# # Print f1, precision, and recall scores
# print("accuracy = ", accuracy_score(y_test_decoded, y_pred_decoded))
# print("precison = ", precision_score(y_test_decoded, y_pred_decoded , average="macro"))
# print("recall = ", recall_score(y_test_decoded, y_pred_decoded , average="macro"))
# print("f1_score =", f1_score(y_test_decoded, y_pred_decoded , average="macro"))



# PLOT TRAINING, VALIDATION AND TEST ACCURACY AND LOSS
#hist_test = np.load('/content/drive/MyDrive/ColabNotebooks/history_test.npy', allow_pickle='TRUE').item()
#history = np.load('/content/drive/MyDrive/ColabNotebooks/history_train.npy', allow_pickle='TRUE').item()
#plot_train_val_test(history, hist_test)


# VIEW FEATURES MAP
model.load_weights("/Users/marco/Desktop/glyphNet/weights/weights.46-0.98.hdf5")

# summarize feature map shapes
# for i in range(len(model.layers)):
#     layer = model.layers[i]
#     print(i, layer.name, layer.output.shape)

c1 = model.layers[1].output
c2 = model.layers[5].output
sc1 = model.layers[9].output
sc2 = model.layers[12].output
sc3 = model.layers[16].output
sc4 = model.layers[19].output
sc5 = model.layers[23].output
sc6 = model.layers[26].output
sc7 = model.layers[30].output
sc8 = model.layers[33].output
sc9 = model.layers[37].output
outputs = [c1,c2,sc1,sc2,sc3,sc4,sc5,sc6,sc7,sc8,sc9]
folder_name = ['c1','c2','sc1','sc2','sc3','sc4','sc5','sc6','sc7','sc8','sc9']

base_url = '/Users/marco/Desktop/D4/'
init_img = 'init-0553_D4.jpg'

# load the image with the required shape
img = load_img(base_url + init_img, grayscale=True, target_size=(100, 100))
# convert the image to an array
img = img_to_array(img)
# expand dimensions so that it represents a single 'sample'
img = expand_dims(img, axis=0)

for i,o in enumerate(outputs):
    # redefine model to output right after the first hidden layer
    model = Model(inputs=model.inputs, outputs=o)
    #model.summary()

    # get feature map for first hidden layer
    feature_maps = model.predict(img)
    feature_maps = feature_maps[0,:,:,:]
    print(feature_maps[:,:,0])

    # for j in range(feature_maps.shape[2]):
    #    cv2.imwrite(base_url + '0-255/'+ folder_name[i] + '/'+ str(j+1) +'.jpg', feature_maps[:,:,j])
    #    plt.imsave(base_url + 'pyplot/'+ folder_name[i] + '/'+ str(j+1) +'.jpg', feature_maps[:,:,j], cmap='gray')


# plot all 64 maps in an 8x8 squares
# square = 23
# ix = 1
# for _ in range(square):
#     for _ in range(square):
#         # specify subplot and turn of axis
#         ax = pyplot.subplot(square, square, ix)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         # plot filter channel in grayscale
#         if ix-1 < feature_maps.shape[3]:
#             pyplot.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
#         ix += 1
# show the figure
# pyplot.show()


