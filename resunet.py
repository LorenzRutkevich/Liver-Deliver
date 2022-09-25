from keras.layers import * 
from keras.utils import plot_model
from keras.models import Model
import keras.backend as K

def res_block(inputs, filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal', use_bias=False):
    x = BatchNormalization()(inputs)
    x = Activation(activation)(x)
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding, kernel_initializer=kernel_initializer, use_bias=use_bias)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Activation(activation)(x)
    shortcut = Conv2D(filters, kernel_size=(1, 1), strides=strides, padding=padding, kernel_initializer=kernel_initializer, use_bias=use_bias)(inputs)
    x = Add()([x, shortcut])
    return x

def res_unet(input_shape=(128, 128, 1), num_classes=1, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal', use_bias=False):
    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv2D(filters, kernel_size, strides=strides, padding=padding, kernel_initializer=kernel_initializer, use_bias=use_bias)(inputs)
    conv1 = res_block(conv1, filters, kernel_size, strides, padding, activation, kernel_initializer, use_bias)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = res_block(pool1, filters*2, kernel_size, strides, padding, activation, kernel_initializer, use_bias)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = res_block(pool2, filters*4, kernel_size, strides, padding, activation, kernel_initializer, use_bias)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = res_block(pool3, filters*8, kernel_size, strides, padding, activation, kernel_initializer, use_bias)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = res_block(pool4, filters*16, kernel_size, strides, padding, activation, kernel_initializer, use_bias)

    # Decoder
    up4 = Conv2DTranspose(filters*8, (2, 2), strides=(2, 2), padding=padding)(conv5)
    up4 = Concatenate(axis=3)([up4, conv4])
    conv6 = res_block(up4, filters*8, kernel_size, strides, padding, activation, kernel_initializer, use_bias)

    up3 = Conv2DTranspose(filters*4, (2, 2), strides=(2, 2), padding=padding)(conv6)
    up3 = Concatenate(axis=3)([up3, conv3])
    conv7 = res_block(up3, filters*4, kernel_size, strides, padding, activation, kernel_initializer, use_bias)

    up2 = Conv2DTranspose(filters*2, (2, 2), strides=(2, 2), padding=padding)(conv7)
    up2 = Concatenate(axis=3)([up2, conv2])
    conv8 = res_block(up2, filters*2, kernel_size, strides, padding, activation, kernel_initializer, use_bias)

    up1 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding=padding)(conv8)
    up1 = Concatenate(axis=3)([up1, conv1])
    conv9 = res_block(up1, filters, kernel_size, strides, padding, activation, kernel_initializer, use_bias)

    out = Conv2D(num_classes, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs, out)
    return model

def residual_block(inputs, filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal', use_bias=False):
    x = BatchNormalization()(inputs)
    x = Activation(activation)(x)
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding, kernel_initializer=kernel_initializer, use_bias=use_bias)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)

    shortcut = Conv2D(filters, kernel_size=(1, 1), strides=strides, padding=padding, kernel_initializer=kernel_initializer, use_bias=use_bias)(inputs)
    x = Add()([x, shortcut])
    return x

def res_unet(input_shape=(128, 128, 1), num_classes=1, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal', use_bias=False, dropout=True):

    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv2D(filters, kernel_size, strides=strides, padding=padding, kernel_initializer=kernel_initializer, use_bias=use_bias)(inputs)
    conv1 = residual_block(conv1, filters, kernel_size, strides, padding, activation, kernel_initializer, use_bias)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    if dropout:
        pool1 = Dropout(0.1)(pool1)


    conv2 = residual_block(pool1, filters*2, kernel_size, strides, padding, activation, kernel_initializer, use_bias)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    if dropout:
        pool2 = Dropout(0.1)(pool2)

    conv3 = residual_block(pool2, filters*4, kernel_size, strides, padding, activation, kernel_initializer, use_bias)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    if dropout:
        pool3 = Dropout(0.1)(pool3)

    conv4 = residual_block(pool3, filters*8, kernel_size, strides, padding, activation, kernel_initializer, use_bias)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    if dropout:
        pool4 = Dropout(0.1)(pool4)

    conv5 = residual_block(pool4, filters*16, kernel_size, strides, padding, activation, kernel_initializer, use_bias)
    if dropout:
     conv5 = Dropout(0.1)(conv5)

    # Decoder
    up6 = Conv2DTranspose(filters*8, (2, 2), strides=(2, 2), padding=padding)(conv5)
    up6 = Concatenate(axis=3)([up6, conv4])
    conv6 = residual_block(up6, filters*8, kernel_size, strides, padding, activation, kernel_initializer, use_bias)
    
    up7 = Conv2DTranspose(filters*4, (2, 2), strides=(2, 2), padding=padding)(conv6)
    up7 = Concatenate(axis=3)([up7, conv3])
    conv7 = residual_block(up7, filters*4, kernel_size, strides, padding, activation, kernel_initializer, use_bias)

    up8 = Conv2DTranspose(filters*2, (2, 2), strides=(2, 2), padding=padding)(conv7)
    up8 = Concatenate(axis=3)([up8, conv2])
    conv8 = residual_block(up8, filters*2, kernel_size, strides, padding, activation, kernel_initializer, use_bias)

    up9 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding=padding)(conv8)
    up9 = Concatenate(axis=3)([up9, conv1])
    conv9 = residual_block(up9, filters, kernel_size, strides, padding, activation, kernel_initializer, use_bias)

    conv10 = Conv2D(num_classes, (1, 1), activation='sigmoid')(conv9)
    
    model = Model(inputs=inputs, outputs=conv10)
    return model

def attention_block(x, g, inter_channel):
    theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1])(x)
    phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1])(g)
    f = Activation('relu')(add([theta_x, phi_g]))
    psi_f = Conv2D(1, [1, 1], strides=[1, 1])(f)
    rate = Activation('sigmoid')(psi_f)
    att_x = multiply([x, rate])
    return att_x

def attention_res_unet(input_shape=(128, 128, 1), num_classes=1, filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal', use_bias=False, dropout=True):
    inputs = Input(input_shape)

    # Down
    conv1 = Conv2D(filters, kernel_size, strides=strides, padding=padding, kernel_initializer=kernel_initializer, use_bias=use_bias)(inputs)
    conv1 = residual_block(conv1, filters, kernel_size, strides, padding, activation, kernel_initializer, use_bias)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    if dropout:
        pool1 = Dropout(0.2)(pool1)


    conv2 = residual_block(pool1, filters*2, kernel_size, strides, padding, activation, kernel_initializer, use_bias)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    if dropout:
        pool2 = Dropout(0.2)(pool2)

    conv3 = residual_block(pool2, filters*4, kernel_size, strides, padding, activation, kernel_initializer, use_bias)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    if dropout:
        pool3 = Dropout(0.2)(pool3)

    conv4 = residual_block(pool3, filters*8, kernel_size, strides, padding, activation, kernel_initializer, use_bias)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    if dropout:
        pool4 = Dropout(0.2)(pool4)

    # Bridge
    conv5 = residual_block(pool4, filters*16, kernel_size, strides, padding, activation, kernel_initializer, use_bias)
    if dropout:
        conv5 = Dropout(0.2)(conv5)

    # Up
    up4 = Conv2DTranspose(filters*8, (2, 2), strides=(2, 2), padding=padding)(conv5)
    up4 = Concatenate(axis=3)([up4, conv4])
    att6 = attention_block(up4, conv4, filters*8)
    att6 = residual_block(att6, filters*8, kernel_size, strides, padding, activation, kernel_initializer, use_bias)


    up3 = Conv2DTranspose(filters*4, (2, 2), strides=(2, 2), padding=padding)(att6)
    up3 = Concatenate(axis=3)([up3, conv3])
    att7 = attention_block(up3, conv3, filters*4)
    att7 = residual_block(att7, filters*4, kernel_size, strides, padding, activation, kernel_initializer, use_bias)


    up2 = Conv2DTranspose(filters*2, (2, 2), strides=(2, 2), padding=padding)(att7)
    up2 = Concatenate(axis=3)([up2, conv2])
    att8 = attention_block(up2, conv2, filters*2)
    att8 = residual_block(att8, filters*2, kernel_size, strides, padding, activation, kernel_initializer, use_bias)


    up1 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding=padding)(conv8)
    up1 = Concatenate(axis=3)([up1, conv1])
    att9 = attention_block(up1, conv1, filters)
    att9 = residual_block(att9, filters, kernel_size, strides, padding, activation, kernel_initializer, use_bias)

    """  glob = GlobalAveragePooling2D()(conv9)
    glob = Dense(128, activation='relu')(glob)
    glob = Dropout(0.1)(glob)
    glob = Dense(64, activation='relu')(glob)
    glob = Dropout(0.1)(glob)
    glob = Dense(32, activation='relu')(glob)
    glob = Dropout(0.1)(glob)
    glob = Dense(16, activation='relu')(glob)
    glob = Dropout(0.1)(glob)
    glob = Dense(8, activation='relu')(glob)
    glob = Dropout(0.1)(glob)
    glob = Dense(4, activation='relu')(glob)
    glob = Dropout(0.1)(glob)
    glob = Dense(2, activation='relu')(glob)
    glob = Dropout(0.1)(glob)
    glob = Dense(1, activation='sigmoid')(glob) """

    out = Conv2D(num_classes, (1, 1), activation='sigmoid')(att9)
    """ out = multiply([out, glob]) """

    model = Model(inputs=inputs, outputs=out)

    return model


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


def gatingsignal(input, out_size, batchnorm=True):
    x = Conv2D(out_size, (1, 1), padding='same')(input)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def attention_block_2(x, gating, inter_shape):
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)
    theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), kernel_initializer='he_normal', padding='same')(x) 
    shape_theta_x = K.int_shape(theta_x)
    phi_g = Conv2D(inter_shape, (1, 1), kernel_initializer='he_normal', padding='same')(gating)
    upsample_g = Conv2DTranspose(inter_shape, (3, 3), strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]), kernel_initializer='he_normal', padding='same')(phi_g)
    concat_xg = add([upsample_g, theta_x])
    act_xg = Activation('relu')(concat_xg)
    psi = Conv2D(1, (1, 1), kernel_initializer='he_normal', padding='same')(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg) 
    upsample_psi = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': shape_x[3]})(upsample_psi)                          
    y = multiply([upsample_psi, x])
    result = Conv2D(shape_x[3], (1, 1), kernel_initializer='he_normal', padding='same')(y)
    attenblock = BatchNormalization()(result)
    return attenblock

def att_res_unet(input_shape=(128,128,1), dropout=0.2, batchnorm=True, filters=16, kernelsize=3, upsample_size=2):
    
    inputs = Input(input_shape)    
    
    # Down
    conv1 = res_conv_block(inputs, kernelsize, filters*1, dropout, batchnorm)
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
    gating5 = gatingsignal(b, filters*8, batchnorm)
    att5 = attention_block_2(conv4, gating5, filters*8)
    up5 = UpSampling2D(size=(upsample_size, upsample_size), data_format="channels_last")(b)
    up5 = concatenate([up5, att5], axis=3)
    up_conv_5 = res_conv_block(up5, kernelsize, filters*8, dropout, batchnorm)
    
    gating4 = gatingsignal(up_conv_5, filters*4, batchnorm)
    att4 = attention_block_2(conv3, gating4, filters*4)
    up4 = UpSampling2D(size=(upsample_size, upsample_size), data_format="channels_last")(up_conv_5)
    up4 = concatenate([up4, att4], axis=3)
    up_conv_4 = res_conv_block(up4, kernelsize, filters*4, dropout, batchnorm)
   
    gating3 = gatingsignal(up_conv_4, filters*2, batchnorm)
    att3 = attention_block_2(conv2, gating3, filters*2)
    up3 = UpSampling2D(size=(upsample_size, upsample_size), data_format="channels_last")(up_conv_4)
    up3 = concatenate([up3, att3], axis=3)
    up_conv_3 = res_conv_block(up3, kernelsize, filters*2, dropout, batchnorm)
    
    gating2 = gatingsignal(up_conv_3, filters*1, batchnorm)
    att2 = attention_block_2(conv1, gating2, filters*1)
    up2 = UpSampling2D(size=(upsample_size, upsample_size), data_format="channels_last")(up_conv_3)
    up2 = concatenate([up2, att2], axis=3)
    up_conv_2 = res_conv_block(up2, kernelsize, filters*1, dropout, batchnorm)

    out = Conv2D(1, kernel_size=(1,1))(up_conv_2)
    out = BatchNormalization(axis=3)(out)
    out = Activation('sigmoid')(out)

    model = Model(inputs, out)
    return model

def double_conv_layer(x, size, dropout, batchnorm):
    conv = Conv2D(size, (3, 3), padding='same')(x)
    if batchnorm:
        conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(size, (3, 3), padding='same')(conv)
    if batchnorm:
        conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    if dropout:
        conv = Dropout(dropout)(conv)
    return conv

def att_res_unet_pp(input_shape=(128,128,1), dropout=0.25, batchnorm=True, filters=16, kernelsize=3, upsample_size=2):
    
    inputs = Input(input_shape)    
    
    # Down
    conv1 = double_conv_layer(inputs, filters*1, dropout, batchnorm)
    conv1 = res_conv_block(conv1, kernelsize, filters*1, dropout, batchnorm)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

    conv2 = double_conv_layer(pool1, filters*2, dropout, batchnorm)
    conv2 = res_conv_block(conv2, kernelsize, filters*2, dropout, batchnorm)
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)

    conv3 = double_conv_layer(pool2, filters*4, dropout, batchnorm)
    conv3 = res_conv_block(conv3, kernelsize, filters*4, dropout, batchnorm)
    pool3 = MaxPooling2D(pool_size=(2,2))(conv3)

    conv4 = double_conv_layer(pool3, filters*8, dropout, batchnorm)
    conv4 = res_conv_block(conv4, kernelsize, filters*8, dropout, batchnorm)
    pool4 = MaxPooling2D(pool_size=(2,2))(conv4)

    # Bridge
    b = double_conv_layer(pool4, filters*16, dropout, batchnorm)

    # Up  
    gating5 = gatingsignal(b, filters*8, batchnorm)
    att5 = attention_block_2(conv4, gating5, filters*8)
    up5 = UpSampling2D(size=(upsample_size, upsample_size), data_format="channels_last")(b)
    up5 = concatenate([up5, att5], axis=3)
    up_conv_5 = double_conv_layer(up5, filters*8, dropout, batchnorm)
    
    gating4 = gatingsignal(up_conv_5, filters*4, batchnorm)
    att4 = attention_block_2(conv3, gating4, filters*4)
    up4 = UpSampling2D(size=(upsample_size, upsample_size), data_format="channels_last")(up_conv_5)
    up4 = concatenate([up4, att4], axis=3)
    up_conv_4 = double_conv_layer(up4, filters*4, dropout, batchnorm)
   
    gating3 = gatingsignal(up_conv_4, filters*2, batchnorm)
    att3 = attention_block_2(conv2, gating3, filters*2)
    up3 = UpSampling2D(size=(upsample_size, upsample_size), data_format="channels_last")(up_conv_4)
    up3 = concatenate([up3, att3],
    axis=3)
    up_conv_3 = double_conv_layer(up3, filters*2, dropout, batchnorm)

    gating2 = gatingsignal(up_conv_3, filters*1, batchnorm)
    att2 = attention_block_2(conv1, gating2, filters*1)
    up2 = UpSampling2D(size=(upsample_size, upsample_size), data_format="channels_last")(up_conv_3)
    up2 = concatenate([up2, att2], axis=3)
    up_conv_2 = double_conv_layer(up2, filters*1, dropout, batchnorm)

    out = Conv2D(1, kernel_size=(1,1))(up_conv_2)
    out = BatchNormalization(axis=3)(out)
    out = Activation('sigmoid')(out)

    model = Model(inputs, out)
    return model

def double_res_att_unet(input_shape=(128,128,1), dropout=0.25, batchnorm=True, filters=16, kernelsize=3, upsample_size=2):
    
    inputs = Input(input_shape)    
    
    # Down
    conv1 = res_conv_block(inputs, kernelsize, filters*1, dropout, batchnorm)
    conv1 = res_conv_block(conv1, kernelsize, filters*1, dropout, batchnorm)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

    conv2 = res_conv_block(pool1, kernelsize, filters*2, dropout, batchnorm)
    conv2 = res_conv_block(conv2, kernelsize, filters*2, dropout, batchnorm)
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)

    conv3 = res_conv_block(pool2, kernelsize, filters*4, dropout, batchnorm)
    conv3 = res_conv_block(conv3, kernelsize, filters*4, dropout, batchnorm)
    pool3 = MaxPooling2D(pool_size=(2,2))(conv3)

    conv4 = res_conv_block(pool3, kernelsize, filters*8, dropout, batchnorm)
    conv4 = res_conv_block(conv4, kernelsize, filters*8, dropout, batchnorm)
    pool4 = MaxPooling2D(pool_size=(2,2))(conv4)

    # Bridge
    b = res_conv_block(pool4, kernelsize, filters*16, dropout, batchnorm)

    # Up  
    gating5 = gatingsignal(b, filters*8, batchnorm)
    att5 = attention_block_2(conv4, gating5, filters*8)
    up5 = UpSampling2D(size=(upsample_size, upsample_size), data_format="channels_last")(b)
    up5 = concatenate([up5, att5], axis=3)
    up_conv_5 = double_conv_layer(up5, filters*8, dropout, batchnorm)
    
    gating4 = gatingsignal(up_conv_5, filters*4, batchnorm)
    att4 = attention_block_2(conv3, gating4, filters*4)
    up4 = UpSampling2D(size=(upsample_size, upsample_size), data_format="channels_last")(up_conv_5)
    up4 = concatenate([up4, att4], axis=3)
    up_conv_4 = double_conv_layer(up4, filters*4, dropout, batchnorm)

    gating3 = gatingsignal(up_conv_4, filters*2, batchnorm)
    att3 = attention_block_2(conv2, gating3, filters*2)
    up3 = UpSampling2D(size=(upsample_size, upsample_size), data_format="channels_last")(up_conv_4)
    up3 = concatenate([up3, att3], axis=3)
    up_conv_3 = double_conv_layer(up3, filters*2, dropout, batchnorm)

    gating2 = gatingsignal(up_conv_3, filters*1, batchnorm)
    att2 = attention_block_2(conv1, gating2, filters*1)
    up2 = UpSampling2D(size=(upsample_size, upsample_size), data_format="channels_last")(up_conv_3)
    up2 = concatenate([up2, att2], axis=3)
    up_conv_2 = double_conv_layer(up2, filters*1, dropout, batchnorm)

    out = Conv2D(1, kernel_size=(1,1))(up_conv_2)
    out = BatchNormalization(axis=3)(out)
    out = Activation('sigmoid')(out)
    
    model = Model(inputs, out)
    return model

if __name__ == '__main__':
    model = double_res_att_unet()
    model.summary()
   # plot_model(model, to_file='model.png', show_shapes=True)


