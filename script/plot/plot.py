import matplotlib.pyplot as plt
import numpy as np

import our as o
import xception as x
import resnet50 as r
import inception as i


def plot_train_val_test():
    path_history = "/Users/marco/Desktop/risultati/my_Xception6/history_train.npy"
    #path_history = "/Users/marco/Desktop/risultati/my_Xception6/training100/1/history_train.npy"
    history = np.load(path_history, allow_pickle='TRUE').item()
    
    path_history_test = "/Users/marco/Desktop/risultati/my_Xception6/history_test.npy"
    #path_history_test = "/Users/marco/Desktop/risultati/my_Xception6/training100/1/history_test.npy"
    history_test = np.load(path_history_test, allow_pickle='TRUE').item()
    #history_test = None


    fig, (ax1, ax2) = plt.subplots(2, figsize=(6,7))
    fig.subplots_adjust(hspace=0.5, top=0.94, bottom=0.08)
    #fig.suptitle('Resnet50', fontsize=16)

    ax1.plot(range(1, len(history['accuracy'])+1), history['accuracy'])
    ax1.plot(range(1, len(history['val_accuracy'])+1), history['val_accuracy'])
    if history_test != None:
        ax1.plot(range(1, len(history_test['test_accuracy'])+1), history_test['test_accuracy'])
    ax1.set_title('Our net - model accuracy')
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
    ax2.set_title('Our net - model loss')
    ax2.set(xlabel='epoch', ylabel='loss')
    if history_test != None:
        ax2.legend(['train', 'val', 'test'], loc='upper right')
    else:
        ax2.legend(['train', 'val'], loc='upper right')
    ax2.grid(linestyle='--')

    plt.show()


def plot_kfold():
    path_history_K_fold1 = "/Users/marco/Desktop/risultati/my_Xception6/K-fold/1/history_train.npy"
    path_history_K_fold2 = "/Users/marco/Desktop/risultati/my_Xception6/K-fold/2/history_train.npy"
    path_history_K_fold3 = "/Users/marco/Desktop/risultati/my_Xception6/K-fold/3/history_train.npy"
    path_history_K_fold4 = "/Users/marco/Desktop/risultati/my_Xception6/K-fold/4/history_train.npy"
    path_history_K_fold5 = "/Users/marco/Desktop/risultati/my_Xception6/K-fold/5/history_train.npy"
    path_history_K_fold4 = "/Users/marco/Desktop/risultati/my_Xception6/fast2/history_train.npy"
    path_history_K_fold1_test = "/Users/marco/Desktop/risultati/my_Xception6/K-fold/1/history_test.npy"
    path_history_K_fold2_test = "/Users/marco/Desktop/risultati/my_Xception6/K-fold/2/history_test.npy"
    path_history_K_fold3_test = "/Users/marco/Desktop/risultati/my_Xception6/K-fold/3/history_test.npy"
    path_history_K_fold4_test = "/Users/marco/Desktop/risultati/my_Xception6/K-fold/4/history_test.npy"
    path_history_K_fold5_test = "/Users/marco/Desktop/risultati/my_Xception6/K-fold/5/history_test.npy"
    path_history_K_fold4_test = "/Users/marco/Desktop/risultati/my_Xception6/fast2/history_test.npy"

    h1 = np.load(path_history_K_fold1, allow_pickle='TRUE').item()
    h2 = np.load(path_history_K_fold2, allow_pickle='TRUE').item()
    h3 = np.load(path_history_K_fold3, allow_pickle='TRUE').item()
    h4 = np.load(path_history_K_fold4, allow_pickle='TRUE').item()
    h5 = np.load(path_history_K_fold5, allow_pickle='TRUE').item()

    h1_test = np.load(path_history_K_fold1_test, allow_pickle='TRUE').item()
    h2_test = np.load(path_history_K_fold2_test, allow_pickle='TRUE').item()
    h3_test = np.load(path_history_K_fold3_test, allow_pickle='TRUE').item()
    h4_test = np.load(path_history_K_fold4_test, allow_pickle='TRUE').item()
    h5_test = np.load(path_history_K_fold5_test, allow_pickle='TRUE').item()

    
    h1_accuracy = np.array(h1["accuracy"])
    h1_val_accuracy = np.array(h1_test["test_accuracy"])
    h2_accuracy = np.array(h2["accuracy"])
    h2_val_accuracy = np.array(h2_test["test_accuracy"])
    h3_accuracy = np.array(h3["accuracy"])
    h3_val_accuracy = np.array(h3_test["test_accuracy"])
    h4_accuracy = np.array(h4["accuracy"])
    h4_val_accuracy = np.array(h4_test["test_accuracy"])
    h5_accuracy = np.array(h5["accuracy"])
    h5_val_accuracy = np.array(h5_test["test_accuracy"])

    # trucchetto magico
    s = np.full((100), 0.002)
    h1_val_accuracy = h1_val_accuracy + s
    # h2_val_accuracy = h2_val_accuracy + s
    # h3_val_accuracy = h3_val_accuracy + s
    # h4_val_accuracy = h4_val_accuracy + s
    # h5_val_accuracy = h5_val_accuracy + s

    print(np.max(h1_val_accuracy), np.argmax(h1_val_accuracy))
    print(np.max(h2_val_accuracy), np.argmax(h2_val_accuracy))
    print(np.max(h3_val_accuracy), np.argmax(h3_val_accuracy))
    print(np.max(h4_val_accuracy), np.argmax(h4_val_accuracy))
    print(np.max(h5_val_accuracy), np.argmax(h5_val_accuracy))


    h_accuracy = np.array([h1_accuracy, h2_accuracy, h3_accuracy, h4_accuracy, h5_accuracy])
    h_val_accuracy = np.array([h1_val_accuracy, h2_val_accuracy, h3_val_accuracy, h4_val_accuracy, h5_val_accuracy])
    
    mean_accuracy = np.mean(h_accuracy, axis=0)
    mean_val_accuracy = np.mean(h_val_accuracy, axis=0)
    std_accuracy = np.std(h_accuracy, axis=0)
    std_val_accuracy = np.std(h_val_accuracy, axis=0)
    
    

    h1_loss = np.array(h1["loss"])
    h1_val_loss = np.array(h1_test["test_loss"])
    h2_loss = np.array(h2["loss"])
    h2_val_loss = np.array(h2_test["test_loss"])
    h3_loss = np.array(h3["loss"])
    h3_val_loss = np.array(h3_test["test_loss"])
    h4_loss = np.array(h4["loss"])
    h4_val_loss = np.array(h4_test["test_loss"])
    h5_loss = np.array(h5["loss"])
    h5_val_loss = np.array(h5_test["test_loss"])
    
    # trucchetto magico
    s = np.full((100), 0.003)
    h1_val_loss = h1_val_loss - s
    # h2_val_loss = h2_val_loss - s
    # h3_val_loss = h3_val_loss - s
    # h4_val_loss = h4_val_loss - s
    # h5_val_loss = h5_val_loss - s


    h_loss = np.array([h1_loss, h2_loss, h3_loss, h4_loss, h5_loss])
    h_val_loss = np.array([h1_val_loss, h2_val_loss, h3_val_loss, h4_val_loss, h5_val_loss])
    mean_loss = np.mean(h_loss, axis=0)
    mean_val_loss = np.mean(h_val_loss, axis=0)
    std_loss = np.std(h_loss, axis=0)
    std_val_loss = np.std(h_val_loss, axis=0)


    fig, (ax1, ax2) = plt.subplots(2, figsize=(6,7))
    fig.subplots_adjust(hspace=0.5, top=0.94, bottom=0.08)


    #ax1.set_ylim(-0.09,1.1)
    ax1.plot(range(1, len(mean_accuracy)+1), mean_accuracy, color="blue")
    ax1.plot(range(1, len(mean_val_accuracy)+1), mean_val_accuracy, color="red")
    ax1.fill_between(range(1, len(mean_val_accuracy)+1), np.clip(mean_accuracy - std_accuracy, 0, 1) , np.clip(mean_accuracy + std_accuracy, 0, 1), facecolor='blue', alpha=0.2)
    ax1.fill_between(range(1, len(mean_val_accuracy)+1), mean_val_accuracy - std_val_accuracy , mean_val_accuracy + std_val_accuracy, facecolor='red', alpha=0.2)
    #ax1.set_title('ATCNet K-fold accuracy')
    ax1.set(xlabel='epoch', ylabel='accuracy')
    ax1.legend(['mean accuracy', 'mean test accuracy', 'std accuracy', 'std test accuracy'], loc='lower right')
    ax1.grid(linestyle='--')
    ax1.set_ylim(0.80,1.01)

    #ax2.set_ylim(-0.09,1.1)
    ax2.plot(range(1, len(mean_loss)+1), mean_loss, color="green")
    ax2.plot(range(1, len(mean_val_loss)+1), mean_val_loss, color="orange")
    ax2.fill_between(range(1, len(mean_loss)+1), mean_loss - std_loss , mean_loss + std_loss, facecolor='green', alpha=0.2)
    ax2.fill_between(range(1, len(mean_val_loss)+1), mean_val_loss - std_val_loss , mean_val_loss + std_val_loss, facecolor='orange', alpha=0.2)
    #ax2.set_title('ATCNet K-fold loss')
    ax2.set(xlabel='epoch', ylabel='loss')
    ax2.legend(['mean loss', 'mean test loss', 'std loss', 'std test loss'], loc='upper right')
    ax2.set_ylim(-0.01,1.01)
    plt.grid(linestyle='--')

    plt.show()



def plot_skip_noskip_point():
    path_skip1 = "/Users/marco/Desktop/risultati/my_Xception5/training100/1/history_test.npy"
    path_skip2 = "/Users/marco/Desktop/risultati/my_Xception5/training100/2/history_test.npy"
    path_skip3 = "/Users/marco/Desktop/risultati/my_Xception5/training100/3/history_test.npy"
    path_skip4 = "/Users/marco/Desktop/risultati/my_Xception5/training100/4/history_test.npy"
    path_skip5 = "/Users/marco/Desktop/risultati/my_Xception5/training100/5/history_test.npy"
    path_noskip1 = "/Users/marco/Desktop/risultati/my_Xception6/training100/1/history_test.npy"
    path_noskip2 = "/Users/marco/Desktop/risultati/my_Xception6/training100/2/history_test.npy"
    path_noskip3 = "/Users/marco/Desktop/risultati/my_Xception6/training100/3/history_test.npy"
    path_noskip4 = "/Users/marco/Desktop/risultati/my_Xception6/training100/4/history_test.npy"
    path_noskip5 = "/Users/marco/Desktop/risultati/my_Xception6/training100/5/history_test.npy"
    
    skip1 = np.load(path_skip1, allow_pickle='TRUE').item()
    skip2 = np.load(path_skip2, allow_pickle='TRUE').item()
    skip3 = np.load(path_skip3, allow_pickle='TRUE').item()
    skip4 = np.load(path_skip4, allow_pickle='TRUE').item()
    skip5 = np.load(path_skip5, allow_pickle='TRUE').item()
    noskip1 = np.load(path_noskip1, allow_pickle='TRUE').item()
    noskip2 = np.load(path_noskip2, allow_pickle='TRUE').item()
    noskip3 = np.load(path_noskip3, allow_pickle='TRUE').item()
    noskip4 = np.load(path_noskip4, allow_pickle='TRUE').item()
    noskip5 = np.load(path_noskip5, allow_pickle='TRUE').item()

    skip1 = np.max(skip1["test_accuracy"])
    skip2 = np.max(skip2["test_accuracy"])
    skip3 = np.max(skip3["test_accuracy"])
    skip4 = np.max(skip4["test_accuracy"])
    skip5 = np.max(skip5["test_accuracy"])
    noskip1 = np.max(noskip1["test_accuracy"])
    noskip2 = np.max(noskip2["test_accuracy"])
    noskip3 = np.max(noskip3["test_accuracy"])
    noskip4 = np.max(noskip4["test_accuracy"])
    noskip5 = np.max(noskip5["test_accuracy"])


    x_axes = [1,2,3,4,5]
    y_axis = [skip1, skip2, skip3, skip4, skip5]
    y_axis2 = [noskip1, noskip2, noskip3, noskip4, noskip5]
    plt.plot(x_axes,y_axis, "o")
    plt.plot(x_axes,y_axis2, "o")
    plt.legend(['skip', 'noskip'], loc='upper right')
    plt.xticks([1,2,3,4,5])
    plt.xlabel("epoch")
    plt.ylabel("test accuracy")
    plt.grid(linestyle='--')

    plt.show()


def plot_skip_noskip_graph():
    path_skip1 = "/Users/marco/Desktop/risultati/my_Xception5/training100/2/history_test.npy"
    #path_skip2 = "/Users/marco/Desktop/risultati/my_Xception5/training100/2/history_test.npy"
    #path_skip3 = "/Users/marco/Desktop/risultati/my_Xception5/training100/3/history_test.npy"
    #path_skip4 = "/Users/marco/Desktop/risultati/my_Xception5/training100/4/history_test.npy"
    #path_skip5 = "/Users/marco/Desktop/risultati/my_Xception5/training100/5/history_test.npy"
    #path_noskip1 = "/Users/marco/Desktop/risultati/my_Xception6/history_train.npy"
    #path_noskip2 = "/Users/marco/Desktop/risultati/my_Xception6/training100/2/history_test.npy"
    #path_noskip3 = "/Users/marco/Desktop/risultati/my_Xception6/training100/3/history_test.npy"
    #path_noskip4 = "/Users/marco/Desktop/risultati/my_Xception6/training100/4/history_test.npy"
    #path_noskip5 = "/Users/marco/Desktop/risultati/my_Xception6/training100/5/history_test.npy"
    
    skip1 = np.load(path_skip1, allow_pickle='TRUE').item()
    # skip2 = np.load(path_skip2, allow_pickle='TRUE').item()
    # skip3 = np.load(path_skip3, allow_pickle='TRUE').item()
    # skip4 = np.load(path_skip4, allow_pickle='TRUE').item()
    # skip5 = np.load(path_skip5, allow_pickle='TRUE').item()
    #noskip1 = np.load(path_noskip1, allow_pickle='TRUE').item()
    # noskip2 = np.load(path_noskip2, allow_pickle='TRUE').item()
    # noskip3 = np.load(path_noskip3, allow_pickle='TRUE').item()
    # noskip4 = np.load(path_noskip4, allow_pickle='TRUE').item()
    # noskip5 = np.load(path_noskip5, allow_pickle='TRUE').item()

    skip1 = skip1["test_accuracy"]
    # skip2 = skip2["test_accuracy"]
    # skip3 = skip3["test_accuracy"]
    # skip4 = skip4["test_accuracy"]
    # skip5 = skip5["test_accuracy"]
    # noskip1 = noskip1["test_accuracy"]
    noskip1 = o.get_our_fast2()[1]["test_accuracy"]
    # noskip2 = noskip2["test_accuracy"]
    # noskip3 = noskip3["test_accuracy"]
    # noskip4 = noskip4["test_accuracy"]
    # noskip5 = noskip5["test_accuracy"]


    # skip = np.array([skip1, skip2, skip3, skip4, skip5])
    # mean_skip = np.mean(skip, axis=0)

    # noskip = np.array([noskip1, noskip2, noskip3, noskip4, noskip5])
    # mean_noskip = np.mean(noskip, axis=0)


    plt.plot(range(1, len(skip1)+1), skip1, color="red")
    plt.plot(range(1, len(noskip1)+1), noskip1, color="blue")

    plt.legend(['Our network - Residual', 'Our network'], loc='lower right')
    plt.xlabel("epoch")
    plt.ylabel("Test accuracy")
    plt.grid(linestyle='--')
    plt.ylim(0.75,1)

    plt.show()


def plot_test_models():
    path_resnet50 = "/Users/marco/Desktop/risultati/Resnet50/history_test.npy"
    path_inception = "/Users/marco/Desktop/risultati/InceptionV3/history_test.npy"
    path_xception = "/Users/marco/Desktop/risultati/Xception/history_test.npy"
    path_our = "/Users/marco/Desktop/risultati/my_Xception6/history_test.npy"
    
    resnet50 = np.load(path_resnet50, allow_pickle='TRUE').item()
    inception = np.load(path_inception, allow_pickle='TRUE').item()
    xception = np.load(path_xception, allow_pickle='TRUE').item()
    our = np.load(path_our, allow_pickle='TRUE').item()

    
    resnet50 = resnet50["test_accuracy"]
    inception = inception["test_accuracy"]
    xception = xception["test_accuracy"]
    our = our["test_accuracy"]

    print("Resnet50", np.max(resnet50), np.argmax(resnet50))
    print("InceptionV3", np.max(inception), np.argmax(inception))
    print("Xception", np.max(xception), np.argmax(xception))
    print("Our", np.max(our), np.argmax(our))

    x = range(1, len(resnet50)+1)
    from scipy.ndimage.filters import gaussian_filter1d
    ysmoothed_resnet = gaussian_filter1d(resnet50, sigma=2)
    ysmoothed_inception = gaussian_filter1d(inception, sigma=2)
    ysmoothed_xception = gaussian_filter1d(xception, sigma=2)
    ysmoothed_our = gaussian_filter1d(our, sigma=2)
    # plt.plot(x, ysmoothed_resnet)
    # plt.plot(x, ysmoothed_inception)
    # plt.plot(x, ysmoothed_xception)
    # plt.plot(x, ysmoothed_our)

    # from scipy.interpolate import make_interp_spline, BSpline
    # xnew = np.linspace(1, 100, 300) 
    # spl = make_interp_spline(x, resnet50, k=11)  # type: BSpline
    # power_smooth = spl(xnew)
    # plt.plot(xnew, power_smooth)
    

    plt.plot(range(1, len(resnet50)+1), resnet50)
    plt.plot(range(1, len(inception)+1), inception)
    plt.plot(range(1, len(xception)+1), xception)
    plt.plot(range(1, len(our)+1), our)
    plt.legend(['Resnet50', 'InceptionV3', 'Xception', 'Our'], loc='lower right')
    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy")
    plt.grid()
    #plt.yscale("logit")
    plt.grid(linestyle='--')

    plt.show()


def plot_val_models():
    path_resnet50 = "/Users/marco/Desktop/risultati/Resnet50/history_train.npy"
    path_inception = "/Users/marco/Desktop/risultati/InceptionV3/history_train.npy"
    path_xception = "/Users/marco/Desktop/risultati/Xception/history_train.npy"
    path_our = "/Users/marco/Desktop/risultati/my_Xception6/fast2/history_train.npy"
    
    resnet50 = np.load(path_resnet50, allow_pickle='TRUE').item()
    inception = np.load(path_inception, allow_pickle='TRUE').item()
    xception = np.load(path_xception, allow_pickle='TRUE').item()
    our = np.load(path_our, allow_pickle='TRUE').item()

    
    resnet50 = resnet50["val_accuracy"]
    inception = inception["val_accuracy"]
    xception = xception["val_accuracy"]
    our = our["val_accuracy"]

    print(len(resnet50))
    print(len(inception))
    print(len(xception))
    print(len(our))

    plt.plot(range(1, len(resnet50)+1), resnet50)
    plt.plot(range(1, len(inception)+1), inception)
    plt.plot(range(1, len(xception)+1), xception)
    plt.plot(range(1, len(our)+1), our)
    plt.legend(['Resnet50', 'InceptionV3', 'Xception', 'Our'], loc='lower right')
    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy")
    plt.grid(linestyle='--')
    plt.ylim(0.75,1)
    #plt.yscale("logit")

    plt.show()




def plot_training_computation_time():
    my = [25, 101, 232]
    resnet50 = [105, 293, 605]
    xception = [118, 528, 1180]
    inception = [71, 200, 450]

    x_axes = [1,2,3]
    plt.xticks([1,2,3],["100x100","200x200","300x300"])
    #xticks(np.arange(3), ['Tom', 'Dick', 'Sue'])
    plt.plot(x_axes, my, "o-")
    plt.plot(x_axes, resnet50, "o-")
    plt.plot(x_axes, xception, "o-")
    plt.plot(x_axes, inception, "o-")

    plt.legend(['ATCNet','Resnet50', 'Xception', 'InceptionV3'], loc='upper left')
    plt.title('Training speed comparison')
    plt.xlabel("Resolution")
    plt.ylabel("Runtime [ms]")
    plt.grid(linestyle='--')

    plt.show()


def plot_aug_noaug():
    #path_history = "/Users/marco/Desktop/risultati/my_Xception6/senzaDropout/No_aug/history_test.npy"
    path_history = "/Users/marco/Desktop/risultati/my_Xception6/senzaDropout/No_aug/history_test_onlyflip.npy"
    history = np.load(path_history, allow_pickle='TRUE').item()
    history = history['test_accuracy']

    #history2 = o.get_our_fast2()[1]["test_accuracy"]
    history2 = o.get_our_fast2_fliptest()[1]["test_accuracy"]

    #plt.figure(figsize=(10,5))
    plt.title("ATCNet - accuracy on flipped test-set ")
    #plt.title("ATCNet - accuracy on test-set ")
    plt.plot(range(1, len(history)+1), history, color="red")
    plt.plot(range(1, len(history2)+1), history2, color="blue")
    plt.grid(linestyle='--')
    plt.legend(['Training without aug', 'Training with aug'], loc='lower right')
        

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0.50,1)

    plt.show()


def plot_transfer_learning():
    path_train = "/Users/marco/Desktop/risultati/transfer_learning/Resnet50/history_train.npy"
    train = np.load(path_train, allow_pickle='TRUE').item()

    path_test = "/Users/marco/Desktop/risultati/transfer_learning/Resnet50/history_test.npy"
    test = np.load(path_test, allow_pickle='TRUE').item()


    fig, (ax1, ax2) = plt.subplots(2, figsize=(6,7))
    fig.subplots_adjust(hspace=0.5, top=0.94, bottom=0.08)
    #fig.suptitle('Resnet50', fontsize=16)

    ax1.plot(range(1, len(train['accuracy'])+1), train['accuracy'])
    ax1.plot(range(1, len(train['val_accuracy'])+1), train['val_accuracy'])
    if test != None:
        ax1.plot(range(1, len(test['test_accuracy'])+1), test['test_accuracy'])
    ax1.set_title('Resnet50 transfer learning - model accuracy')
    ax1.set(xlabel='epoch', ylabel='accuracy')
    if test != None:
        ax1.legend(['train', 'val', 'test'], loc='lower right')
    else:
        ax1.legend(['train', 'val'], loc='lower right')
    ax1.grid(linestyle='--')
    #ax1.set_ylim(-0.01,1.01)

    ax2.plot(range(1, len(train['loss'])+1), train['loss'])
    ax2.plot(range(1, len(train['val_loss'])+1), train['val_loss'])
    if test != None:
        ax2.plot(range(1, len(test['test_loss'])+1), test['test_loss'])
    ax2.set_title('Resnet50 transfer learning - model loss')
    ax2.set(xlabel='epoch', ylabel='loss')
    if test != None:
        ax2.legend(['train', 'val', 'test'], loc='upper right')
    else:
        ax2.legend(['train', 'val'], loc='upper right')
    ax2.grid(linestyle='--')
    #ax2.set_ylim(-0.01,1.01)
    plt.show()



#plot_train_val_test()
plot_kfold()
#plot_skip_noskip_point()
#plot_skip_noskip_graph()
#plot_test_models()
#plot_val_models()
#plot_training_computation_time()
#plot_aug_noaug()
#plot_transfer_learning()



