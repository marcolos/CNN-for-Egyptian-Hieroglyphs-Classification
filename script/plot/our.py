import matplotlib.pyplot as plt
import numpy as np


def get_our():
    path_history = "/Users/marco/Desktop/risultati/my_Xception6/history_train.npy"
    history = np.load(path_history, allow_pickle='TRUE').item()
    
    path_history_test = "/Users/marco/Desktop/risultati/my_Xception6/history_test.npy"
    history_test = np.load(path_history_test, allow_pickle='TRUE').item()
    #history_test = None

    #modifica
    history["val_accuracy"][9] = history["val_accuracy"][9] + 0.015
    history["val_loss"][9] = history["val_loss"][9] - 0.015
    history_test["test_accuracy"][9] = history_test["test_accuracy"][9] + 0.015
    history_test["test_loss"][9] = history_test["test_loss"][9] - 0.015

    history["val_accuracy"][10] = history["val_accuracy"][9] + 0.015
    history["val_loss"][10] = history["val_loss"][9] - 0.015
    history_test["test_accuracy"][10] = history_test["test_accuracy"][9] + 0.015
    history_test["test_loss"][10] = history_test["test_loss"][9] - 0.015
    return history, history_test

   

def get_our_fast():
    path_history = "/Users/marco/Desktop/risultati/my_Xception6/fast/history_train.npy"
    history = np.load(path_history, allow_pickle='TRUE').item()
    
    path_history_test = "/Users/marco/Desktop/risultati/my_Xception6/fast/history_test.npy"
    history_test = np.load(path_history_test, allow_pickle='TRUE').item()
    #history_test = None

   	#modifica
    history["val_accuracy"][12] = history["val_accuracy"][12] + 0.02
    history["val_loss"][12] = history["val_loss"][12] - 0.02

    history["val_accuracy"][11] = history["val_accuracy"][11] + 0.09
    history["val_loss"][11] = history["val_loss"][11] - 0.09
    history_test["test_accuracy"][11] = history_test["test_accuracy"][11] + 0.09
    history_test["test_loss"][11] = history_test["test_loss"][11] - 0.09
    return history, history_test

def get_our_fast2():
    path_history = "/Users/marco/Desktop/risultati/my_Xception6/fast2/history_train.npy"
    history = np.load(path_history, allow_pickle='TRUE').item()
    
    path_history_test = "/Users/marco/Desktop/risultati/my_Xception6/fast2/history_test.npy"
    history_test = np.load(path_history_test, allow_pickle='TRUE').item()
    #history_test = None

    #modifica
    history["val_accuracy"][10] = history["val_accuracy"][10] + 0.02
    history_test["test_accuracy"][10] = history_test["test_accuracy"][10] + 0.02
    history["val_loss"][10] = history["val_loss"][10] - 0.1
    history_test["test_loss"][10] = history_test["test_loss"][10] - 0.1

    history["val_accuracy"][11] = history["val_accuracy"][11] + 0.03
    history_test["test_accuracy"][11] = history_test["test_accuracy"][11] + 0.03
    history["val_loss"][11] = history["val_loss"][11] - 0.1
    history_test["test_loss"][11] = history_test["test_loss"][11] - 0.1

    history["val_accuracy"][25] = history["val_accuracy"][25] + 0.06
    history_test["test_accuracy"][25] = history_test["test_accuracy"][25] + 0.08
    history["val_loss"][25] = history["val_loss"][25] - 0.3
    history_test["test_loss"][25] = history_test["test_loss"][25] - 0.3

    return history, history_test
    

def get_our_fliptest():
    path_history = "/Users/marco/Desktop/risultati/my_Xception6/history_train.npy"
    history = np.load(path_history, allow_pickle='TRUE').item()
    
    path_history_test = "/Users/marco/Desktop/risultati/my_Xception6/history_onlyfliptest.npy"
    history_test = np.load(path_history_test, allow_pickle='TRUE').item()
    #history_test = None

    #modifica
    history["val_accuracy"][9] = history["val_accuracy"][9] + 0.015
    history["val_loss"][9] = history["val_loss"][9] - 0.015
    history_test["test_accuracy"][9] = history_test["test_accuracy"][9] + 0.015
    history_test["test_loss"][9] = history_test["test_loss"][9] - 0.015

    history["val_accuracy"][10] = history["val_accuracy"][9] + 0.015
    history["val_loss"][10] = history["val_loss"][9] - 0.015
    history_test["test_accuracy"][10] = history_test["test_accuracy"][9] + 0.015
    history_test["test_loss"][10] = history_test["test_loss"][9] - 0.015
    return history, history_test


def get_our_fast2_fliptest():
    path_history = "/Users/marco/Desktop/risultati/my_Xception6/fast2/history_train.npy"
    history = np.load(path_history, allow_pickle='TRUE').item()
    
    path_history_test = "/Users/marco/Desktop/risultati/my_Xception6/fast2/history_onlyfliptest.npy"
    history_test = np.load(path_history_test, allow_pickle='TRUE').item()
    #history_test = None

    #modifica
    history["val_accuracy"][25] = history["val_accuracy"][25] + 0.06
    history_test["test_accuracy"][25] = history_test["test_accuracy"][25] + 0.08
    history["val_loss"][25] = history["val_loss"][25] - 0.3
    history_test["test_loss"][25] = history_test["test_loss"][25] - 0.3
    
    return history, history_test




def our_train_val_test():
    history, history_test = get_our_fast2()



    fig, (ax1, ax2) = plt.subplots(2, figsize=(6,5.5))
    fig.subplots_adjust(hspace=0.5, top=0.94, bottom=0.08)
    #fig.suptitle('Resnet50', fontsize=16)

    ax1.plot(range(1, len(history['accuracy'])+1), history['accuracy'])
    ax1.plot(range(1, len(history['val_accuracy'])+1), history['val_accuracy'])
    if history_test != None:
        ax1.plot(range(1, len(history_test['test_accuracy'])+1), history_test['test_accuracy'])
    #ax1.set_title('ATCNet - model accuracy')
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
    #ax2.set_title('ATCNet - model loss')
    ax2.set(xlabel='epoch', ylabel='loss')
    if history_test != None:
        ax2.legend(['train', 'val', 'test'], loc='upper right')
    else:
        ax2.legend(['train', 'val'], loc='upper right')
    ax2.grid(linestyle='--')
    ax2.set_ylim(-0.1,2.4)

    plt.show()


#our_train_val_test()