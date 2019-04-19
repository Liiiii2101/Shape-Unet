import tensorflow as tf
from keras.models import Model, load_model
import keras.backend as K
from keras.models import Sequential
from keras.layers import (Input, Conv2D, Conv2DTranspose,
                            MaxPooling2D, Concatenate, UpSampling2D,
                            Conv3D, Conv3DTranspose, MaxPooling3D,
                            UpSampling3D,Reshape,Flatten,Dense)
from keras import optimizers as opt
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization,Activation,Input,Dropout,Subtract
import keras
from keras.layers import Lambda
from keras.layers.merge import concatenate, add


def Euclidean_loss(y_true, y_pred):
    return  K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

def norm(outputs,vtrans):
    ''' Self-defined layer'''
    #outputs = K.transpose(outputs)
    outputs = K.dot(outputs,vtrans)
    #outputs = K.transpose(outputs)

    return outputs


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def deconv2d_block(input_tensor, n_filters, kernel_size=2, batchnorm=True):
    # first layer
    x = Conv2DTranspose(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x



def get_unet(vtrans,shape_parameter=True):

    input_im=Input((64, 64, 1))

    c1 = conv2d_block(input_im, 64)
    c1_2 = conv2d_block(c1, 64)
    p1 = MaxPooling2D(pool_size=(2,2),strides=2,padding='same') (c1_2)
  
    c2 = conv2d_block(p1, 128)
    c2_2 = conv2d_block(c2, 128)
    p2 = MaxPooling2D(pool_size=(2,2),strides=2,padding='same') (c2_2)
  
    c3 = conv2d_block(p2, 256)
    c3_2 = conv2d_block(c3, 256)
    p3 = MaxPooling2D(pool_size=(2,2),strides=2,padding='same') (c3_2)
  
    c4 = conv2d_block(p3, 512)
    c4_2 = conv2d_block(c4, 512)
  
    c5 = conv2d_block(c4_2, 1024)
    c5_2 = conv2d_block(c5, 1024)
  
    c6 = conv2d_block(c5_2, 512)
    c6_2 = conv2d_block(c6, 512)
    c6_2 = concatenate([c6_2, c4_2])
  
    c7 = deconv2d_block(c6_2,512)
    c7 = UpSampling2D(size=(2,2))(c7)
    c8 = conv2d_block(c7, 256)
    c8_2 = conv2d_block(c8, 256)
    c8_2 = concatenate([c8_2,c3_2])
  
    c9 = deconv2d_block(c8_2,256)
    c9 = UpSampling2D(size=(2,2))(c9)
    c10 = conv2d_block(c9, 128)
    c10_2 = conv2d_block(c10, 128)
    c10_2 = concatenate([c10_2,c2_2])
  
    c11 = deconv2d_block(c10_2,128)
    c11 = UpSampling2D(size=(2,2))(c11)
    c12 = conv2d_block(c11, 64)
    c12_2 = conv2d_block(c12, 64)
    c12_2 = concatenate([c12_2,c1_2])
  
    c13 = Conv2D(1, kernel_size=(3, 3),padding="same")(c12_2)
  
    if shape_parameter:

  	  r1 = Flatten()(c13)
  	  r2 = Lambda(norm,arguments={'vtrans':vtrans})(r1)

    else:
  	  r2 = c13
  
    model = Model(inputs=[input_im], outputs=r2)

    return model








