import matplotlib.pyplot as plt
import numpy as np
import our as o
import xception as x
import resnet50 as r
import inception as i


def plot_test_models():
    
	resnet50 = r.get_resnet()[1]["test_accuracy"]
	inception = i.get_inception()[1]["test_accuracy"]
	xception = x.get_xception()[1]["test_accuracy"]
	our = o.get_our_fast2()[1]["test_accuracy"]

	print("Resnet50", np.max(resnet50), np.argmax(resnet50))
	print("InceptionV3", np.max(inception), np.argmax(inception))
	print("Xception", np.max(xception), np.argmax(xception))
	print("ATCNet", np.max(our), np.argmax(our))

    # x = range(1, len(resnet50)+1)
    # from scipy.ndimage.filters import gaussian_filter1d
    # ysmoothed_resnet = gaussian_filter1d(resnet50, sigma=2)
    # ysmoothed_inception = gaussian_filter1d(inception, sigma=2)
    # ysmoothed_xception = gaussian_filter1d(xception, sigma=2)
    # ysmoothed_our = gaussian_filter1d(our, sigma=2)
    # plt.plot(x, ysmoothed_resnet)
    # plt.plot(x, ysmoothed_inception)
    # plt.plot(x, ysmoothed_xception)
    # plt.plot(x, ysmoothed_our)
    #---------------------------------
    # from scipy.interpolate import make_interp_spline, BSpline
    # xnew = np.linspace(1, 100, 300) 
    # spl = make_interp_spline(x, resnet50, k=11)  # type: BSpline
    # power_smooth = spl(xnew)
    # plt.plot(xnew, power_smooth)
    
	#plt.title("Accuracy trends on the test-set")
	plt.plot(range(1, len(our)+1), our, color="blue")
	plt.plot(range(1, len(resnet50)+1), resnet50, color="red")
	plt.plot(range(1, len(inception)+1), inception, color="green")
	plt.plot(range(1, len(xception)+1), xception, color="orange")
	plt.legend([ 'GlyphNet', 'Resnet50', 'InceptionV3', 'Xception'], loc='lower right')
	plt.xlabel("Epoch")
	plt.ylabel("accuracy")
	plt.grid()
	#plt.yscale("logit")
	plt.grid(linestyle='--')
	plt.ylim(0.75,1)
	plt.show()


def plot_test_two_model():
    
	our = o.get_our_fast2()[1]["test_accuracy"]
	#resnet50 = r.get_resnet()[1]["test_accuracy"]
	#inception = i.get_inception()[1]["test_accuracy"]
	vs = x.get_xception()[1]["test_accuracy"]
    
	plt.plot(range(1, len(our)+1), our, color="blue")
	#plt.plot(range(1, len(resnet50)+1), resnet50, color="red")
	#plt.plot(range(1, len(inception)+1), inception, color="green")
	plt.plot(range(1, len(vs)+1), vs, color="orange")

	#plt.legend(['Our', 'Resnet50', 'InceptionV3', 'Xception'], loc='lower right')
	plt.legend(['Our', 'Xception'], loc='lower right')
	plt.xlabel("Epoch")
	plt.ylabel("Test accuracy")
	plt.grid()
	#plt.yscale("logit")
	plt.ylim(0.75,1)
	plt.grid(linestyle='--')
	

	plt.show()



plot_test_models()
#plot_test_two_model()