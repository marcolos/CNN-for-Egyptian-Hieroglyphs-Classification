import matplotlib.pyplot as plt
import numpy as np


def get_inception():
    path_history = "/Users/marco/Desktop/risultati/InceptionV3/history_train.npy"
    history = np.load(path_history, allow_pickle='TRUE').item()
    
    path_history_test = "/Users/marco/Desktop/risultati/InceptionV3/history_test.npy"
    history_test = np.load(path_history_test, allow_pickle='TRUE').item()
    #history_test = None

    return history, history_test




def inception_train_val_test():
    history, history_test = get_inception()

    fig, (ax1, ax2) = plt.subplots(2, figsize=(6,5.5))
    fig.subplots_adjust(hspace=0.5, top=0.94, bottom=0.08)
    #fig.suptitle('Resnet50', fontsize=16)

    ax1.plot(range(1, len(history['accuracy'])+1), history['accuracy'])
    ax1.plot(range(1, len(history['val_accuracy'])+1), history['val_accuracy'])
    if history_test != None:
        ax1.plot(range(1, len(history_test['test_accuracy'])+1), history_test['test_accuracy'])
    #ax1.set_title('InceptionV3 - model accuracy')
    ax1.set(xlabel='epoch', ylabel='accuracy')
    if history_test != None:
        ax1.legend(['train', 'val', 'test'], loc='lower right')
    else:
        ax1.legend(['train', 'val'], loc='lower right')
    ax1.grid(linestyle='--')
    ax1.set_ylim(0.75,1.01)

    ax2.plot(range(1, len(history['loss'])+1), history['loss'])
    ax2.plot(range(1, len(history['val_loss'])+1), history['val_loss'])
    if history_test != None:
        ax2.plot(range(1, len(history_test['test_loss'])+1), history_test['test_loss'])
    #ax2.set_title('InceptionV3 - model loss')
    ax2.set(xlabel='epoch', ylabel='loss')
    if history_test != None:
        ax2.legend(['train', 'val', 'test'], loc='upper right')
    else:
        ax2.legend(['train', 'val'], loc='upper right')
    ax2.grid(linestyle='--')
    ax2.set_ylim(-0.1,2.4)

    plt.show()


#inception_train_val_test()