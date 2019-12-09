import sys
import cv2
import PIL
import skimage
import numpy as np
from PIL import Image
from keras import models
from keras.models import load_model
import matplotlib.pyplot as plt
from skimage.transform import resize

from utilities import predict, scale_up
from helpers import BilinearUpSampling2D
from utilities import depth_loss_function

plasma = plt.get_cmap('plasma')
video = cv2.VideoCapture(0)

def get_img_arr(image):
    im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (640, 480))
    x = np.clip(np.asarray(im, dtype=float) / 255, 0, 1)
    return x

def display_single_image(output, inputs=None, is_colormap=True):
    import matplotlib.pyplot as plt

    plasma = plt.get_cmap('plasma')
    imgs = []
    imgs.append(inputs)

    # #rescale output
    out_min = np.min(output)
    out_max = np.max(output)
    output = output - out_min
    outputs = output / out_max

    if is_colormap:
        rescaled = outputs[:, :, 0]
        pred_x = plasma(rescaled)[:, :, :3]
        imgs.append(pred_x)

    img_set = np.hstack(imgs)
    return img_set

if __name__ == "__main__":
	args = sys.argv[ 1: ]
	if (len(args) <= 0) :
		sys.exit( 0 )
	model_name = args[0]
	
	custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': depth_loss_function }
	# Load the saved model
	print('Loading model...')
	# model_name = 'nyu.h5'
	model = load_model(model_name, custom_objects=custom_objects, compile=False)
	print('\nModel loaded ({0}).'.format(model_name))
	
	while True:
	    (runnin, frame) = video.read()
	    im = Image.fromarray(frame, 'RGB')
	    img_array = np.array(im)
	    img_arr = get_img_arr(img_array)
	    output = scale_up(2, predict(model, img_arr, batch_size=1))
	    pred = output.reshape(output.shape[1], output.shape[2], 1)
	    img_set = display_single_image(pred, img_arr)
	    cv2.imshow('Capturing', img_set)
	    key = cv2.waitKey(1)
	    if key == ord('q'):
	        break
	video.release()
	cv2.destroyAllWindows()