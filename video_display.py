import cv2
import PIL
import numpy as np
from PIL import Image
from keras import models
from keras.models import load_model
import matplotlib.pyplot as plt
import skimage
import sys
from skimage.transform import resize

from utils import predict
from layers import BilinearUpSampling2D
from loss import depth_loss_function

plasma = plt.get_cmap('plasma')
video = cv2.VideoCapture(0)

def load_images_with_resize(image_files):
    loaded_images = []
    for file in image_files:
        im = Image.open( file )
        im = im.resize((640, 480), PIL.Image.ANTIALIAS)
        x = np.clip(np.asarray(im, dtype=float) / 255, 0, 1)
        loaded_images.append(x)
    return np.stack(loaded_images, axis=0)

def getImage(output):
	rescaled = output[:,:,0]
	rescaled = rescaled - np.min(rescaled)
	rescaled = rescaled / np.max(rescaled)
	img = plasma(rescaled)[:,:,:3]
	return img

# Input images
# inputs = load_images_with_resize( glob.glob(args.input) )
# print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))
if __name__ == "__main__":
	args = sys.argv[ 1: ]
	if (len(args) <= 0) :
		sys.exit( 0 )
	model_name = args[0]
	
	custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': depth_loss_function}
	print('Loading model...')

	#Load the saved model
	model = load_model(model_name, custom_objects=custom_objects, compile=False)
	print('\nModel loaded ({0}).'.format(model_name))
	
	while True:
		_, frame = video.read()

		im = Image.fromarray(frame, 'RGB')

		im = im.resize((640, 480))
		img_array = np.array(im)
	
		#Convert the captured frame into RGB
		# im = Image.fromarray(frame, 'RGB')
		# im = Image.open( file )
		print("here")
		# im = frame.resize((640, 480), PIL.Image.ANTIALIAS)
		x = np.clip(np.asarray(img_array, dtype=float) / 255, 0, 1)
		# img = np.stack([x], axis=0)
		output = predict(model, x)
		frame = getImage(output.copy())
		cv2.imshow("Capturing", output)
		key=cv2.waitKey(1)
		if key == ord('q'):
			break
	video.release()
	cv2.destroyAllWindows()