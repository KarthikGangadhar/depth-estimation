import os, sys, glob, time, pathlib

from keras import applications
from keras.models import Model, load_model
from keras.layers import Input, InputLayer, Conv2D, Activation, LeakyReLU, Concatenate
from layers import BilinearUpSampling2D
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras.utils.vis_utils import plot_model

# Kerasa / TensorFlow
from utilities import get_nyu_train_test_data, load_test_data, depth_loss_function
from callbacks import get_nyu_callbacks


# Define upsampling layer
def upproject(base_model, tensor, filters, name, concat_with):
    up_i = BilinearUpSampling2D((2, 2), name=name+'_upsampling2d')(tensor)
    up_i = Concatenate(name=name+'_concat')([up_i, base_model.get_layer(concat_with).output]) # Skip connection
    up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convA')(up_i)
    up_i = LeakyReLU(alpha=0.2)(up_i)
    up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convB')(up_i)
    up_i = LeakyReLU(alpha=0.2)(up_i)
    return up_i

def create_model():
    
    # Encoder Layers
    print('Loading base model (DenseNet)..')
    base_model = applications.DenseNet121(input_shape=(None, None, 3), include_top=False, weights='imagenet')
    print(base_model.summary())
    print('Base model loaded.')

    # Starting point for decoder
    base_model_output_shape = base_model.layers[-1].output.shape
    print(" Output shape is: ", base_model_output_shape)

    # Layer freezing?
    for layer in base_model.layers: layer.trainable = True
    
    # Starting half number of decoder filters
    
    decode_filters = int(int(base_model_output_shape[-1])/2)

    decoder = Conv2D(filters=decode_filters, kernel_size=1, padding='same', input_shape=base_model_output_shape, name='conv2')(base_model.output)
    decoder = upproject(base_model, decoder, int(decode_filters/2), 'up1', concat_with='pool3_pool')
    decoder = upproject(base_model, decoder, int(decode_filters/4), 'up2', concat_with='pool2_pool')
    decoder = upproject(base_model, decoder, int(decode_filters/8), 'up3', concat_with='pool1')
    decoder = upproject(base_model, decoder, int(decode_filters/16), 'up4', concat_with='conv1/relu')
    if False: decoder = upproject(base_model, decoder, int(decode_filters/32), 'up5', concat_with='input_1')
    
    # Extract depths (final layer)
    conv3 = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', name='conv3')(decoder)
    
    # Create the model
    model = Model(inputs=base_model.input, outputs=conv3)
    print('Model created.')
    
    return (base_model,model)

def train(batch_size = 5 , epochs = 5, lr = 0.0001):
    batch_size = int(args[0])
    epochs = int(args[1]) 
    lr = float(args[2])
    print("batch_size = {0} , epochs = {1}, lr = {2}".format(batch_size,epochs, lr))

    #creates encoder and decoder model
    base_model , model = create_model()

    train_generator, test_generator = get_nyu_train_test_data( batch_size )

    # Training session details
    runPath = os.path.join(os.getcwd(),'models',str(int(time.time())))
    pathlib.Path(runPath).mkdir(parents=True, exist_ok=True)
    print('Output: ' + runPath)

    # Optimizer
    optimizer = Adam(lr=lr, amsgrad=True)

    model.compile(loss=depth_loss_function, optimizer=optimizer)

    print('Ready for training!\n')

    # callbacks = []
    # callbacks = get_nyu_callbacks(model, base_model, train_generator, test_generator, load_test_data(), runPath)

    # Start training
    model.fit_generator(train_generator, callbacks=None, validation_data=test_generator, epochs=epochs, shuffle=True)

    # Save the final trained model:
    basemodel.save(runPath + '/model.h5')
    pass

if __name__ == "__main__":
    args = sys.argv[ 1: ]
    if (len(args) <= 0) :
        sys.exit( 0 )
    
    batch_size= int(args[0]), 
    epochs=int(args[1]), 
    lr=float(args[2]),
    train(batch_size,epochs,lr)