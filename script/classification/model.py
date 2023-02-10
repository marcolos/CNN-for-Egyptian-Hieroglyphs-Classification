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


def ATCNet(shape, n_classes):  

    # INPUT BLOCK

    input = Input(shape=shape)
    x = Conv2D(64, (3, 3), padding='same', use_bias=False, name='input_block_conv1')(input)
    x = BatchNormalization(name='input_block_conv1_bn')(x)
    x = MaxPooling2D((3, 3), strides=(2,2), padding="same", name = "input_block_conv1_pool")(x)  #aggiunto
    x = Activation('relu', name='input_block_conv1_act')(x)  #aggiunto
    x = Conv2D(64, (3, 3), padding='same', use_bias=False, name='input_block_conv2')(x)
    x = BatchNormalization(name='input_block_conv2_bn')(x)
    x = MaxPooling2D((3, 3), strides=(2,2), padding="same", name = "input_block_conv2_pool")(x)  #aggiunto
    x = Activation('relu', name='input_block_conv2_act')(x)


    # MIDDLE BLOCKS

    # MIDDLE BLOCK 1
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='middle_block1_sepconv1')(x)
    x = BatchNormalization(name='middle_block1_sepconv1_bn')(x)
    x = Activation('relu', name='middle_block1_sepconv1_act')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='middle_block1_sepconv2')(x)
    x = BatchNormalization(name='middle_block1_sepconv2_bn')(x)
    x = MaxPooling2D((3, 3), strides=(2,2), padding="same", name='middle_block1_sepconv2_pool')(x)
    x = Activation('relu', name='middle_block1_sepconv2_act')(x)

    # MIDDLE BLOCK 2
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='middle_block2_sepconv1')(x)
    x = BatchNormalization(name='middle_block2_sepconv1_bn')(x)
    x = Activation('relu', name='middle_block2_sepconv1_act')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='middle_block2_sepconv2')(x)
    x = BatchNormalization(name='middle_block2_sepconv2_bn')(x)
    x = MaxPooling2D((3, 3), strides=(2,2), padding="same", name='middle_block2_sepconv2_pool')(x)
    x = Activation('relu', name='middle_block2_sepconv2_act')(x)

    # MIDDLE BLOCK 3
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='middle_block3_sepconv1')(x)
    x = BatchNormalization(name='middle_block3_sepconv1_bn')(x)
    x = Activation('relu', name='middle_block3_sepconv1_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='middle_block3_sepconv2')(x)
    x = BatchNormalization(name='middle_block3_sepconv2_bn')(x)
    x = MaxPooling2D((3, 3), strides=(2,2), padding="same", name='middle_block3_sepconv2_pool')(x)
    x = Activation('relu', name='middle_block3_sepconv2_act')(x)

    # MIDDLE BLOCK 4
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='middle_block4_sepconv1')(x)
    x = BatchNormalization(name='middle_block4_sepconv1_bn')(x)
    x = Activation('relu', name='middle_block4_sepconv1_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='middle_block4_sepconv2')(x)
    x = BatchNormalization(name='middle_block4_sepconv2_bn')(x)
    x = MaxPooling2D((3, 3), strides=(2,2), padding="same", name='middle_block4_sepconv2_pool')(x)
    x = Activation('relu', name='middle_block4_sepconv2_act')(x)

    # EXIT BLOCK 
    x = SeparableConv2D(512, (3, 3), padding='same', use_bias=False, name='exit_block_sepconv')(x)
    x = BatchNormalization(name='exit_block_sepconv_bn')(x)
    x = Activation('relu', name='exit_block_sepconv_act')(x)

    # TOP
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = Dropout(0.2, name="dropout")(x)
    output = Dense(n_classes, activation='softmax', name='predictions', kernel_regularizer=regularizers.l2(0.01))(x)
    model = Model(input, output, name="ATCNet")
    return model