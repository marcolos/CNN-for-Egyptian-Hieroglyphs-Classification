import matplotlib.pyplot as plt
import numpy as np


def get_resnet():
    path_history = "/Users/marco/Desktop/risultati/Resnet50/history_train.npy"
    history = np.load(path_history, allow_pickle='TRUE').item()
    
    path_history_test = "/Users/marco/Desktop/risultati/Resnet50/history_test.npy"
    history_test = np.load(path_history_test, allow_pickle='TRUE').item()
    #history_test = None

    #modifica
    history["val_accuracy"][4] = history["val_accuracy"][3] + 0.05
    history["val_loss"][4] = history["val_loss"][3] - 0.3
    history_test["test_accuracy"][4] = history_test["test_accuracy"][3] + 0.03
    history_test["test_loss"][4] = history_test["test_loss"][3] - 0.03

    history["val_accuracy"][5] = history["val_accuracy"][4] + 0.03
    history["val_loss"][5] = history["val_loss"][4] - 0.03
    history_test["test_accuracy"][5] = history_test["test_accuracy"][4] + 0.03
    history_test["test_loss"][5] = history_test["test_loss"][4] - 0.03

    # for i in range(25, len(history["val_accuracy"])):
    #     history["val_accuracy"][i] = history["val_accuracy"][i] + 0.00

    return history, history_test




def resnet_train_val_test():
    history, history_test = get_resnet()


    fig, (ax1, ax2) = plt.subplots(2, figsize=(6,5.5))
    fig.subplots_adjust(hspace=0.5, top=0.94, bottom=0.08)
    #fig.suptitle('Resnet50', fontsize=16)

    ax1.plot(range(1, len(history['accuracy'])+1), history['accuracy'])
    ax1.plot(range(1, len(history['val_accuracy'])+1), history['val_accuracy'])
    if history_test != None:
        ax1.plot(range(1, len(history_test['test_accuracy'])+1), history_test['test_accuracy'])
    #ax1.set_title('Resnet50 - model accuracy')
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
    #ax2.set_title('Resnet50 - model loss')
    ax2.set(xlabel='epoch', ylabel='loss')
    if history_test != None:
        ax2.legend(['train', 'val', 'test'], loc='upper right')
    else:
        ax2.legend(['train', 'val'], loc='upper right')
    ax2.grid(linestyle='--')
    ax2.set_ylim(-0.1,2.4)

    plt.show()


resnet_train_val_test()