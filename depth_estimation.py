import os, sys, glob, time, pathlib

import keras
from keras import applications
from keras.models import Model, load_model
from keras.layers import Input, InputLayer, Conv2D, Activation, LeakyReLU, Concatenate, MaxPooling2D, Dropout, UpSampling2D, concatenate
from helpers import BilinearUpSampling2D
from keras.optimizers import Adam

# Kerasa / TensorFlow
from utilities import get_nyu_train_test_data, load_test_data, depth_loss_function



def create_model():
        
    print('Loading base model (UNet)..')
    inputs = Input(shape=(None, None, 3))
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    conv10 = Conv2D(1, 1)(conv8)

    model = Model(input = inputs, output = conv10)
    model.summary()
    print(model.layers[-1].output.shape)
    return model

def train(args, batch_size = 5 , epochs = 5, lr = 0.0001):
    
    if batch_size.__class__ == tuple:
        batch_size = batch_size[0]
    if epochs.__class__ == tuple:
        epochs = epochs[0]
    if lr.__class__ == tuple:
        lr = lr[0]                
    
    print("batch_size = {0} , epochs = {1}, lr = {2}".format(batch_size,epochs, lr))

    #creates encoder and decoder model
    model = create_model()

    train_generator, test_generator = get_nyu_train_test_data( batch_size )

    # Training session details
    runPath = os.path.join(os.getcwd(),'models',str(int(time.time())))
    pathlib.Path(runPath).mkdir(parents=True, exist_ok=True)
    print('Output: ' + runPath)

    # Optimizer
    optimizer = Adam(lr=lr, amsgrad=True)

    model.compile(loss=depth_loss_function, optimizer=optimizer)

    # Start training
    model.fit_generator(train_generator, callbacks=None, validation_data=test_generator, epochs=epochs, shuffle=True)

    # Save the final trained model:
    print('Model Save Began')
    model.save(runPath + '/unetmodel.h5')
    print('Model (unetmodel.h5) has been Saved!!')
    pass

if __name__ == "__main__":
    args = sys.argv[ 1: ]
    if (len(args) <= 0) :
        sys.exit( 0 )
    
    batch_size = int(args[0]), 
    epochs = int(args[1]), 
    lr = float(args[2]),
    train(args, batch_size, epochs, lr)