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

from utils import predict, scale_up
from layers import BilinearUpSampling2D
from loss import depth_loss_function

plasma = plt.get_cmap('plasma')
video = cv2.VideoCapture(0)

# def load_images_with_resize(image_files):
#     loaded_images = []
#     for file in image_files:
#         im = Image.open( file )
#         im = im.resize((640, 480), PIL.Image.ANTIALIAS)
#         x = np.clip(np.asarray(im, dtype=float) / 255, 0, 1)
#         loaded_images.append(x)
#     return np.stack(loaded_images, axis=0)

# def getImage(output):
# 	rescaled = output[:,:,0]
# 	rescaled = rescaled - np.min(rescaled)
# 	rescaled = rescaled / np.max(rescaled)
# 	img = plasma(rescaled)[:,:,:3]
# 	return img
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

    ##rescale output
    out_min = np.min(output)
    out_max = np.max(output)
    output = output - out_min
    outputs = output/out_max

    if is_colormap:
        rescaled = outputs[:, :, 0]
        pred_x = plasma(rescaled)[:, :, :3]
        imgs.append(pred_x)

    img_set = np.hstack(imgs)

    return img_set

    

# Input images
# inputs = load_images_with_resize( glob.glob(args.input) )
# print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))
# if __name__ == "__main__":
# 	args = sys.argv[ 1: ]
# 	if (len(args) <= 0) :
# 		sys.exit( 0 )
# 	model_name = args[0]
	
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': depth_loss_function}
print('Loading model...')
#Load the saved model
model = load_model(model_name, custom_objects=custom_objects, compile=False)
print('\nModel loaded ({0}).'.format(model_name))

while True:
	_, frame = video.read()
	im = Image.fromarray(frame, 'RGB')
	# im = im.resize((640, 480))
	img_array = np.array(im)
	if key == ord('q'):
		break
video.release()
cv2.destroyAllWindows()