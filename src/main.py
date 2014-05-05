import os

import matplotlib.pyplot as plt

from sklearn.externals import joblib

import ImageCropper
import DiagramAnalyzer
from ComponentClassifier import ComponentClassifier

class System:

	def __init__(self, clf_fn, nbins, scaler_fn):
		self.data_dir = os.path.join(os.path.abspath('../'),'data/')
		self.images_dir = os.path.join(self.data_dir, 'TestImages/')
		self.clf = joblib.load(os.path.join(self.data_dir, clf_fn))
		self.nbins = nbins
		self.scaler = joblib.load(os.path.join(self.data_dir, scaler_fn))

	def run(self, image):
		"""

		1. Smartly crop around the full circuit schematic DiagramAnalyzer
			inputs:
				image: raw image file 
			outpus:
				cropped: black and white cropped image

		2. Split image into "Segment" objects
			inputs:
				cropped: black and white cropped image
			outputs:
				segments: list of Segment objects
		
		For each segment in segments:
			
			3. Find components along a segment and crop around them
				inputs: 
					segment: a segment object 
				outputs:
					component_images: ordered list of PIL images representing
									  the components from a line segment
			
			4. Resistor vs Capacitor Component Classifier
				inputs:
					component_images: ordered list of PIL images representing 
					                  the components from a line segment
				outputs:
					stores ordered list of the names of the components in the
					.component_id_list attr
			
			5. Draw a beautified digital equivalent of the segment
				inputs: 
					segment: Segment object, with attr componet_id_list 
				outputs:
					Stores beautiful image in .image attr

		6. Draw and Show the entire circuit diagram  digitally and beautified
			inputs:
				segments: list of segment objects, now with the beautified 
						  image of the segments stored in the .image attr

		"""
		cropped = ImageCropper.smart_crop(image)
		segments = DiagramAnalyzer.get_segments(cropped)
		
		for segment in segments:
			# FInd component along a segment and crop around them
			plt.imshow(segment.image, cmap='gray')
			plt.show()
			component_images = DiagramAnalyzer.component_finder(segment)

			# Resistor vs Capacitor Component Classifier
			component_id_list = ComponentClassifier.predict(
									component_images, 
									self.clf, 
									self.nbins,
									self.scaler)

			print component_id_list
			segment.finding_components(component_id_list)

			# Draw a beautified digital equivalent of the segment
			DiagramAnalyzer.draw_segment(segment)

		# Draw and Show the entire circuit diagram digitally and beautified
		DiagramAnalyzer.final_draw(segments)


if __name__ == '__main__':
	SmarterBoard = System(clf_fn='SVC_scaled_moments_nbins23_INTERP_CUBIC.pkl',
						  nbins=23, 
						  scaler_fn='Scaler_nbins23.pkl')

	fn = 'intersection-test.jpg'
	image = ImageCropper.open_file(os.path.join(SmarterBoard.images_dir, fn))
	
	SmarterBoard.run(image)