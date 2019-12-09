# import numpy as np
# from PIL import Image

# def DepthNorm(x, maxDepth):
#     return maxDepth / x

# def predict(model, images, minDepth=10, maxDepth=1000, batch_size=2):
#     # Support multiple RGBs, one RGB image, even grayscale 
#     if len(images.shape) < 3: images = np.stack((images,images,images), axis=2)
#     if len(images.shape) < 4: images = images.reshape((1, images.shape[0], images.shape[1], images.shape[2]))
#     # Compute predictions
#     predictions = model.predict(images, batch_size=batch_size)
#     # Put in expected range
#     return np.clip(DepthNorm(predictions, maxDepth=1000), minDepth, maxDepth) / maxDepth

# def scale_up(scale, images):
#     from skimage.transform import resize
#     scaled = []
    
#     for i in range(len(images)):
#         img = images[i]
#         output_shape = (scale * img.shape[0], scale * img.shape[1])
#         scaled.append( resize(img, output_shape, order=1, preserve_range=True, mode='reflect', anti_aliasing=True ) )

#     return np.stack(scaled)

