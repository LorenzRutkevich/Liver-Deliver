from keras.layers import *
from keras.models import *
import keras.backend as K


def res_conv_block(x, kernelsize, filters, dropout, batchnorm=True):
    conv1 = Conv2D(filters, (kernelsize, kernelsize), kernel_initializer='he_normal', padding='same')(x)
    if batchnorm is True:
        conv1 = BatchNormalization(axis=3)(conv1)
    conv1 = Activation('relu')(conv1)    
    conv2 = Conv2D(filters, (kernelsize, kernelsize), kernel_initializer='he_normal', padding='same')(conv1)
    if batchnorm is True:
        conv2 = BatchNormalization(axis=3)(conv2)
        conv2 = Activation("relu")(conv2)
    if dropout > 0:
        conv2 = Dropout(dropout)(conv2)
    short = Conv2D(filters, kernel_size=(1, 1), kernel_initializer='he_normal', padding='same')(x)
    if batchnorm is True:
        short = BatchNormalization(axis=3)(short)
    short = Activation("relu")(short)
    respath = add([short, conv2])       
    return respath

def attention_block_2(x, gating, inter_shape):
    shape_x = K.int_shape(x)
    shape_gating = K.int_shape(gating)
    theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)
    shape_theta_x = K.int_shape(theta_x)
    phi_g = Conv2D(inter_shape, (1, 1), padding='same')(gating)
    upsample_g = UpSampling2D(size=(shape_theta_x[1] // shape_gating[1], shape_theta_x[2] // shape_gating[2]))(phi_g)
    concat_xg = add([upsample_g, theta_x])
    act_xg = Activation('relu')(concat_xg)
    psi = Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    upsample_psi = UpSampling2D(size=(shape_x[1] // shape_theta_x[1], shape_x[2] // shape_theta_x[2]))(sigmoid_xg)
    y = multiply([upsample_psi, x])
    result = Conv2D(shape_x[3], (1, 1), padding='same')(y)
    result_bn = BatchNormalization()(result)
    return result_bn

def recurrent_block(x, filters, dropout, batchnorm=True, t =2):
    for i in range(t):
        x = res_conv_block(x, 3, filters, dropout, batchnorm)
    return x


    
def new(input_shape=(128,128,1), dropout=0.25, batchnorm=True, filters=16, kernelsize=3, upsample_size=2):
    
    inputs = Input(input_shape)    
    
    # Down
    conv1 = res_conv_block(inputs, kernelsize, filters, dropout, batchnorm)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

    conv2 = res_conv_block(pool1, kernelsize, filters*2, dropout, batchnorm)
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)

    conv3 = res_conv_block(pool2, kernelsize, filters*4, dropout, batchnorm)
    pool3 = MaxPooling2D(pool_size=(2,2))(conv3)

    conv4 = res_conv_block(pool3, kernelsize, filters*8, dropout, batchnorm)
    pool4 = MaxPooling2D(pool_size=(2,2))(conv4)

    # Bridge
    b = res_conv_block(pool4, kernelsize, filters*16, dropout, batchnorm)

    # Up
    att = attention_block_2(conv4, b, filters*8)
    up1 = concatenate([UpSampling2D(size=(upsample_size, upsample_size))(b), att], axis=3)
    conv5 = res_conv_block(up1, kernelsize, filters*8, dropout, batchnorm)

    att = attention_block_2(conv3, conv5, filters*4)
    up2 = concatenate([UpSampling2D(size=(upsample_size, upsample_size))(conv5), att], axis=3)
    conv6 = res_conv_block(up2, kernelsize, filters*4, dropout, batchnorm)

    att = attention_block_2(conv2, conv6, filters*2)
    up3 = concatenate([UpSampling2D(size=(upsample_size, upsample_size))(conv6), att], axis=3)
    conv7 = res_conv_block(up3, kernelsize, filters*2, dropout, batchnorm)

    att = attention_block_2(conv1, conv7, filters*1)
    up4 = concatenate([UpSampling2D(size=(upsample_size, upsample_size))(conv7), att], axis=3)
    conv8 = res_conv_block(up4, kernelsize, filters*1, dropout, batchnorm)

    # Output
    conv9 = Conv2D(1, (1,1), activation='sigmoid')(conv8)

    model = Model(inputs=[inputs], outputs=[conv9])

    return model
