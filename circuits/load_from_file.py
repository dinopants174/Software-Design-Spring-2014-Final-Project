from bw_componentrecognition import *

DATA_DIR = os.path.join(os.path.abspath('.'), 'misc/')
NBINS = 6

if __name__ == '__main__':
	clf = load_clf("bw_rescap.pkl")
	fn = DATA_DIR + 'resistor1.jpg'
	image = loadImage(fn)
	X = loadImageFeatures(fn, NBINS)
	y_pred = clf.predict(X)

	print y_pred

	components = ["Resistor" if label == True else "Capacitor" for label in y_pred]
	print components
	plt.imshow(image, cmap='gray')
	plt.show()


