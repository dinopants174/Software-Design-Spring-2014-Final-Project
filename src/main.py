import os

from sklearn.externals import joblib

# Import scripts used in our system including crop, straighten, etc
# *** HERE ***
# *** HERE ***
from ComponentClassifier import ComponentClassifier

class System:

	data_dir = os.path.join(os.path.abspath('../'),'data/')

	ml_classifier = joblib.load(self.data_dir+'randomforest_rawpix_nbins6.pkl')
	nbins = 6

	@staticmethod
	def run():
		"""
		1. crop processes
		2. straighten processes
		3. resistor vs capacitor component classifier
			inputs:
				component_images: ordered list of PIL images representing the 
								  components from a line segment
			outputs:
				component_names: ordered list of the names of the components
								 (e.g. ['resistor', 'capacitor', 'resistor'])
		4. rest of the process ...
		"""
		component_names = ComponentClassifier.predict(component_images, 
												self.ml_classifier, self.nbins)

if __name__ == '__main__':
	System.run()