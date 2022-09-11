from keras.layers import *
from keras.models import *


def easy_conv(x, filters, kernel_size, strides=1, padding='same', activation='relu', name=None):
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding, activation=activation, name=name)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    return x


def residual_block(x, filters, kernel_size, strides=1, padding='same', activation='relu', name=None, skips=[]):
    x = easy_conv(x, filters, kernel_size, strides=strides, padding=padding, activation=activation, name=name)
    for skip in skips:
        x = Add()([x, skip])
    return x


def concat_block(x, filters, kernel_size, strides=1, padding='same', activation='relu', name=None, skips=[]):
    x = easy_conv(x, filters, kernel_size, strides=strides, padding=padding, activation=activation, name=name)
    x = Concatenate()([x, *skips])
    return x


def up_concat(x, xskip):
    u = UpSampling2D((2, 2))(x)
    c = concatenate([u, xskip])
    return c


def easy_conv_batch(x, filters, kernel_size=3, strides=1, padding='same', activation='relu'):
    conv = Conv2D(filters, kernel_size, strides=strides, padding=padding, activation=activation)(x)
    conv = BatchNormalization()(conv)
    return conv

def transpose_block(x, filters, kernel_size=3, strides=1, padding='same', activation='relu'):
    conv = Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding, activation=activation)(x)
    conv = BatchNormalization()(conv)
    return conv

def nested_block(x, filters, kernel_size=3, strides=1, padding='same', activation='relu', name=None):
    conv = Conv2D(filters, kernel_size, strides=strides, padding=padding, activation=activation)(x)
    conv = BatchNormalization()(conv)
    conv = Conv2D(filters, kernel_size, strides=strides, padding=padding, activation=activation)(conv)
    conv = BatchNormalization()(conv)
    conv = Conv2D(filters, kernel_size, strides=strides, padding=padding, activation=activation)(conv)
    conv = BatchNormalization()(conv)
    conv = Conv2D(filters, kernel_size, strides=strides, padding=padding, activation=activation)(conv)
    conv = BatchNormalization()(conv)
    return conv

def attention_block(x, g, inter_channel):
    theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1])(x)
    phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1])(g)
    f = Activation('relu')(add([theta_x, phi_g]))
    psi_f = Conv2D(1, [1, 1], strides=[1, 1])(f)
    rate = Activation('sigmoid')(psi_f)
    att_x = multiply([x, rate])
    return att_x


def double_conv(x, filters, kernel_size=3, strides=1, padding='same', activation='relu'):
    conv = Conv2D(filters, kernel_size, strides=strides, padding=padding, activation=activation)(x)
    conv = BatchNormalization()(conv)
    conv = Conv2D(filters, kernel_size, strides=strides, padding=padding, activation=activation)(conv)
    conv = BatchNormalization()(conv)
    return conv


def recurrent_block(x, filters, kernel_size=3, strides=1, padding='same', activation='relu'):
    conv = Conv2D(filters, kernel_size, strides=strides, padding=padding, activation=activation)(x)
    conv = BatchNormalization()(conv)
    conv = Conv2D(filters, kernel_size, strides=strides, padding=padding, activation=activation)(conv)
    conv = BatchNormalization()(conv)
    conv = Conv2D(filters, kernel_size, strides=strides, padding=padding, activation=activation)(conv)
    conv = BatchNormalization()(conv)
    return conv


def up_concat_dropout(x, down, filters, dropout=0.2):
    up = UpSampling2D((2, 2))(x)
    up = Conv2D(filters, (2, 2), padding='same', activation='relu')(up)
    up = Dropout(dropout)(up)
    up = concatenate([up, down])
    return up

def recurrent_residual_block(x, filters, kernel_size=3, strides=1, padding='same', activation='relu', name=None, skips=[]):
    x = recurrent_block(x, filters, kernel_size, strides=strides, padding=padding, activation=activation, name=name)
    for skip in skips:
        x = Add()([x, skip])
    return x


def NestedUnet(input_shape=(128, 128, 1), num_classes=1):
    inputs = Input(shape=input_shape)
    x = inputs
    skips = []
    filters = [32, 64, 128, 256, 512]
    for i, f in enumerate(filters):
        x = nested_block(x, f, 3, name=f'nested{i}')
        skips.append(x)
        x = MaxPool2D(2)(x)
    x = nested_block(x, 1024, 3, name='nested5')
    skips = skips[::-1]
    filters = filters[::-1]
    for i, f in enumerate(filters):
        x = UpSampling2D(2)(x)
        x = concat_block(x, f, 3, name=f'concat{i}', skips=[skips[i]])
    x = Conv2D(1, 1, activation='sigmoid')(x)
    model = Model(inputs, x)
    return model
    
def ResUnet(input_shape=(128, 128, 1), num_classes=1):
    inputs = Input(shape=input_shape)
    x = inputs
    skips = []
    filters = [32, 64, 128, 256, 512]
    for i, f in enumerate(filters):
        x = residual_block(x, f, 3, name=f'residual{i}')
        skips.append(x)
        x = MaxPool2D(2)(x)
    x = residual_block(x, 1024, 3, name='residual5')
    skips = skips[::-1]
    filters = filters[::-1]
    for i, f in enumerate(filters):
        x = UpSampling2D(2)(x)
        x = concat_block(x, f, 3, name=f'concat{i}', skips=[skips[i]])
    x = Conv2D(1, 1, activation='sigmoid')(x)
    model = Model(inputs, x)
    return model


def Unet(input_shape=(128, 128, 1), num_classes=1):
    input = Input(input_shape)
    filters = [32, 64, 128, 256, 512]
    # Encoder
    e0 = input
    e1 = easy_conv_batch(e0, filters[0])
    e2 = easy_conv_batch(MaxPooling2D((2, 2))(e1), filters[1])
    e3 = easy_conv_batch(MaxPooling2D((2, 2))(e2), filters[2])
    e4 = easy_conv_batch(MaxPooling2D((2, 2))(e3), filters[3])
    e5 = easy_conv_batch(MaxPooling2D((2, 2))(e4), filters[4])
    # Bridge
    b0 = Dropout(0.5)(e5)
    b1 = easy_conv_batch(b0, filters[4], kernel_size=3)
    # Decoder
    u1 = up_concat(b1, e4)
    d1 = easy_conv_batch(u1, filters[4])
    u2 = up_concat(d1, e3)
    d2 = easy_conv_batch(u2, filters[3])
    u3 = up_concat(d2, e2)
    d3 = easy_conv_batch(u3, filters[2])
    u4 = up_concat(d3, e1)
    d4 = easy_conv_batch(u4, filters[1])
    # Output
    output = Conv2D(num_classes, (1, 1), padding='same', activation='sigmoid')(d4)
    model = Model(input, output)
    return model


def AttUnet(input_shape=(128, 128, 1), num_classes=1):
    input = Input(input_shape)
    filters = [32, 64, 128, 256, 512]
    # Encoder
    e0 = input
    e1 = easy_conv_batch(e0, filters[0])
    e2 = easy_conv_batch(MaxPooling2D((2, 2))(e1), filters[1])
    e3 = easy_conv_batch(MaxPooling2D((2, 2))(e2), filters[2])
    e4 = easy_conv_batch(MaxPooling2D((2, 2))(e3), filters[3])
    e5 = easy_conv_batch(MaxPooling2D((2, 2))(e4), filters[4])
    # Bridge
    b0 = Dropout(0.5)(e5)
    b1 = easy_conv_batch(b0, filters[4], kernel_size=3)
    # Decoder
    u1 = UpSampling2D((2, 2))(b1)
    a1 = attention_block(u1, e4, int(filters[4] / 2))
    c1 = concatenate([a1, e4])
    d1 = easy_conv_batch(c1, filters[4])
    u2 = UpSampling2D((2, 2))(d1)
    a2 = attention_block(u2, e3, int(filters[3] / 2))
    c2 = concatenate([a2, e3])
    d2 = easy_conv_batch(c2, filters[3])
    u3 = UpSampling2D((2, 2))(d2)
    a3 = attention_block(u3, e2, int(filters[2] / 2))
    c3 = concatenate([a3, e2])
    d3 = easy_conv_batch(c3, filters[2])
    u4 = UpSampling2D((2, 2))(d3)
    a4 = attention_block(u4, e1, int(filters[1] / 2))
    c4 = concatenate([a4, e1])
    d4 = easy_conv_batch(c4, filters[1])
    # Output
    output = Conv2D(num_classes, (1, 1), padding='same', activation='sigmoid')(d4)
    model = Model(input, output)
    return model


def RecurrentUnet(input_shape=(128, 128, 1), num_classes=1):
    input = Input(input_shape)
    filters = [32, 64, 128, 256, 512]
    # Encoder
    e0 = input
    e1 = double_conv(e0, filters[0])
    e2 = double_conv(MaxPooling2D((2, 2))(e1), filters[1])
    e3 = double_conv(MaxPooling2D((2, 2))(e2), filters[2])
    e4 = double_conv(MaxPooling2D((2, 2))(e3), filters[3])
    e5 = double_conv(MaxPooling2D((2, 2))(e4), filters[4])
    # Bridge
    b0 = Dropout(0.5)(e5)
    b1 = recurrent_block(b0, filters[4], kernel_size=3)
    # Decoder
    u1 = up_concat(b1, e4)
    d1 = double_conv(u1, filters[4])
    u2 = up_concat(d1, e3)
    d2 = double_conv(u2, filters[3])
    u3 = up_concat(d2, e2)
    d3 = double_conv(u3, filters[2])
    u4 = up_concat(d3, e1)
    d4 = double_conv(u4, filters[1])
    # Output
    output = Conv2D(num_classes, (1, 1), padding='same', activation='sigmoid')(d4)
    model = Model(input, output)
    return model


def double_unet(input_shape=(128, 128, 1), num_classes=1):
    input = Input(input_shape)
    filters = [32, 64, 128, 256, 512]
    # Encoder
    e0 = input
    e1 = double_conv(e0, filters[0])
    e2 = double_conv(MaxPooling2D((2, 2))(e1), filters[1])
    e3 = double_conv(MaxPooling2D((2, 2))(e2), filters[2])
    e4 = double_conv(MaxPooling2D((2, 2))(e3), filters[3])
    e5 = double_conv(MaxPooling2D((2, 2))(e4), filters[4])
    # Bridge
    b0 = Dropout(0.5)(e5)
    b1 = double_conv(b0, filters[4], kernel_size=3)
    # Decoder
    u1 = up_concat(b1, e4)
    d1 = double_conv(u1, filters[4])
    u2 = up_concat(d1, e3)
    d2 = double_conv(u2, filters[3])
    u3 = up_concat(d2, e2)
    d3 = double_conv(u3, filters[2])
    u4 = up_concat(d3, e1)
    d4 = double_conv(u4, filters[1])
    # Output
    output = Conv2D(num_classes, (1, 1), padding='same', activation='sigmoid')(d4)
    model = Model(input, output)
    return model


def small_unet(inputs=(128, 128, 1), num_classes=1):
    inputs = Input(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = up_concat_dropout(conv5, conv4, 256)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = up_concat_dropout(conv6, conv3, 128)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = up_concat_dropout(conv7, conv2, 64)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = up_concat_dropout(conv8, conv1, 32)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(num_classes, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    return model

def Crop_Unet(input_shape=(128,128,1), target_size=(128,128)):
    inputs = Input(input_shape)
    filters = [32,64,128,256,512]
    x = inputs
    for i, f in enumerate(filters):
        x = Conv2D(f, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x = Conv2D(f, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
        if i < len(filters)-1:
            x = Conv2D(f, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
            x = Conv2D(f, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
            x = MaxPooling2D(pool_size=(2,2))(x)    
    x = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)

    x = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)

    x = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)

    x = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)

    x = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)

   
    x = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)

    
    x = Conv2DTranspose(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2DTranspose(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2DTranspose(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2DTranspose(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2DTranspose(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2DTranspose(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2DTranspose(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2DTranspose(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2DTranspose(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2DTranspose(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    out = Conv2D(1, 1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=out)
    
    

    return model

