import h5py
import keras
import random
from keras.layers import Input, Dense, Permute, Reshape
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D, GlobalMaxPooling2D
from keras.layers import Activation, Dropout, Flatten, concatenate, Maximum, Lambda, Multiply, Add, add, multiply, SpatialDropout2D, GaussianDropout, AlphaDropout
from keras.optimizers import Adam, Adadelta, RMSprop
from keras.regularizers import l2
from keras import backend as K

import math
use_bias = True
taylor = 5


def share_stream(x_shape,outputNum=256):
    input = Input(x_shape)
    # x_a = Transformer(d_k, frames)(input)
    # # x_a = Permute((3,1,2))(x_a)
    # print(x_a._keras_shape)
    x_a=input	
    # x_a = Dropout(0.1)(x_a)

    conv1 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1),  padding='same',
                   use_bias=use_bias,trainable=True)(x_a)
    # conv1 = BatchNormalization(axis=3, epsilon=1.001e-5)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1),  padding='same',
                   use_bias=use_bias,trainable=True)(conv1)
    # conv1 = BatchNormalization(axis=3, epsilon=1.001e-5)(conv1)
    x_a = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1),  padding='same',
                   use_bias=use_bias,trainable=True)(x_a)
    x_a = add([x_a,conv1])
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),  padding='same',
                   use_bias=use_bias,trainable=True)(x_a)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1),  padding='same',
                   use_bias=use_bias)(conv1)
    x_a = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1),  padding='same',
                   use_bias=use_bias,trainable=True)(x_a)
    x_a = add([x_a,conv1])
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),  padding='same',
                   use_bias=use_bias,trainable=True)(x_a)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1),  padding='same',
                   use_bias=use_bias,trainable=True)(conv1)
    x_a = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1),  padding='same',
                   use_bias=use_bias,trainable=True)(x_a)
    x_a = add([x_a,conv1])
    attention = Conv2D(filters=32, kernel_size=(5, 5), activation = 'sigmoid', strides=(1, 1),  padding='same',
                   use_bias=use_bias,trainable=True)(x_a)
    x_a = multiply([x_a,attention])
    x_a = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x_a)
    # x_a = Dropout(0.3)(x_a)
    # x_a = SpatialDropout2D(0.2)(x_a)
	
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),  padding='same',
                   use_bias=use_bias,trainable=True)(x_a)
    # conv1 = BatchNormalization(axis=3, epsilon=1.001e-5)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1),  padding='same',
                   use_bias=use_bias,trainable=True)(conv1)
    # conv1 = BatchNormalization(axis=3, epsilon=1.001e-5)(conv1)
    x_a = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1),  padding='same',
                   use_bias=use_bias,trainable=True)(x_a)
    x_a = add([x_a,conv1])
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),  padding='same',
                   use_bias=use_bias,trainable=True)(x_a)
    # conv1 = BatchNormalization(axis=3, epsilon=1.001e-5)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1),  padding='same',
                   use_bias=use_bias,trainable=True)(conv1)
    # conv1 = BatchNormalization(axis=3, epsilon=1.001e-5)(conv1)
    x_a = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1),  padding='same',
                   use_bias=use_bias,trainable=True)(x_a)
    x_a = add([x_a,conv1])
    attention = Conv2D(filters=64, kernel_size=(5, 5), activation = 'sigmoid', strides=(1, 1),  padding='same',
                   use_bias=use_bias,trainable=True)(x_a)
    x_a = multiply([x_a,attention])
    x_a = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x_a)
    # x_a = SpatialDropout2D(0.2)(x_a)
    # x_a = Dropout(0.5)(x_a)
    # x_a = GaussianDropout(0.3)(x_a)


    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same',
                   use_bias=use_bias,trainable=True)(x_a)
    x2Shape=x._keras_shape
    x2 = Conv2D(filters = x2Shape[3]*taylor, kernel_size=(3, 3),padding = 'same',trainable=True)(Lambda(lambda x: x*10000000., output_shape=x2Shape[1:])(Activation("tanh")(Lambda(lambda x: x/10000000., output_shape=x2Shape[1:])(x))))
    x2 = Reshape((x2Shape[1], x2Shape[2], taylor, x2Shape[3]))(x2)
    temp = Multiply()([Lambda(lambda x: x[:,:,:,1,:], output_shape=x2Shape[1:])(x2),x])
    for jj in range(2,taylor):
        temp = Add()([Multiply()([Lambda(lambda x: x[:,:,:,jj,:], output_shape=x2Shape[1:])(x2),Lambda(lambda x: (x ** jj)/math.factorial(jj), output_shape=x2Shape[1:])(x)]),temp])
    temp = Add()([Lambda(lambda x: x[:,:,:,0,:], output_shape=x2Shape[1:])(x2),temp])
    conv2 =Lambda(lambda x: x*10000000., output_shape=x2Shape[1:])(Activation("tanh")(Lambda(lambda x: x/10000000., output_shape=x2Shape[1:])(temp)))
    conv2 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1),  padding='same',
                   use_bias=use_bias,trainable=True)(conv2)
    x_a = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1),  padding='same',
                   use_bias=use_bias,trainable=True)(x_a)
    x_a = add([x_a,conv2])
    x_a = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x_a)
    # x_a = SpatialDropout2D(0.2)(x_a)
    # x_a = GaussianDropout(0.3)(x_a)
	

    conv3 = Conv2D(filters=outputNum, kernel_size=(3, 3), strides=(1, 1), padding='same',
                   use_bias=use_bias,trainable=True)(x_a)
    # conv3 = BatchNormalization(axis=3, epsilon=1.001e-5)(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(filters=outputNum, kernel_size=(1, 1),strides=(1, 1), padding='same',
                   use_bias=use_bias,trainable=True)(conv3)
    # conv3 = BatchNormalization(axis=3, epsilon=1.001e-5)(conv3)
    x_a = Conv2D(filters=outputNum, kernel_size=(1, 1),strides=(1, 1),  padding='same',
                   use_bias=use_bias,trainable=True)(x_a)
    x_a = add([x_a,conv3])
    attention = Conv2D(filters=outputNum, kernel_size=(3, 3), activation = 'sigmoid', strides=(1, 1),  padding='same',
                   use_bias=use_bias,trainable=True)(x_a)
    x_a = multiply([x_a,attention])

    x_a = GlobalMaxPooling2D()(x_a)

    shared_layer = Model(input, x_a)
    return shared_layer


def model(input_shape,input_shape2):
    up_0 = Input(shape=input_shape, name='up_stream_0')
    up_1 = Input(shape=input_shape, name='up_stream_1')
    down_0 = Input(shape=input_shape, name='down_stream_0')
    down_1 = Input(shape=input_shape, name='down_stream_1')
    down_02 = Input(shape=input_shape, name='down_stream_02')
    down_12 = Input(shape=input_shape, name='down_stream_12')
    down_2 = Input(shape=input_shape2, name='down_stream_2')
    down_3 = Input(shape=input_shape2, name='down_stream_3')

    up_stream = share_stream(x_shape=input_shape,outputNum=128)
    down_stream = share_stream(x_shape=input_shape,outputNum=128)
    # down_stream2 = share_stream(x_shape=input_shape)
    down_stream3 = share_stream(x_shape=input_shape2,outputNum=128)

    up_feature_0 = up_stream(up_0)
    up_feature_1 = up_stream(up_1)
    down_feature_0 = down_stream(down_0)
    down_feature_1 = down_stream(down_1)
    down_feature_02 = down_stream(down_02)
    down_feature_12 = down_stream(down_12)
    down_feature_2 = down_stream3(down_2)
    down_feature_3 = down_stream3(down_3)

    up_feature = Maximum()([up_feature_0, up_feature_1])
    down_feature = Maximum()([down_feature_0, down_feature_1])
    down_feature2 = Maximum()([down_feature_02, down_feature_12])
    down_feature3 = Maximum()([down_feature_2, down_feature_3])
    # up_feature = concatenate([up_feature_0, up_feature_1])
    # down_feature = concatenate([down_feature_0, down_feature_1])
    # down_feature2 = concatenate([down_feature_02, down_feature_12])
    # down_feature3 = concatenate([down_feature_2, down_feature_3])

    feature = concatenate([up_feature, down_feature, down_feature2, down_feature3])
    # feature = GaussianDropout(0.3)(feature)

    x = Dense(units=256, use_bias=True,trainable=True)(feature)
    x2Shape=x._keras_shape
    x2 = Dense(16,trainable=True)(Lambda(lambda x: x*10000000., output_shape=x2Shape[1:])(Activation("tanh")(Lambda(lambda x: x/10000000., output_shape=x2Shape[1:])(x))))
    x2Shape2=x2._keras_shape
    x2 = Dense(256*taylor,trainable=True)(Lambda(lambda x: x*10000000., output_shape=x2Shape2[1:])(Activation("tanh")(Lambda(lambda x: x/10000000., output_shape=x2Shape2[1:])(x2))))
    x2 = Reshape((256, taylor))(x2)
    temp = Multiply()([Lambda(lambda x: x[:,:,1], output_shape=x2Shape[1:])(x2),x])
    for jj in range(2,taylor):
        temp = Add()([Multiply()([Lambda(lambda x: x[:,:,jj], output_shape=x2Shape[1:])(x2),Lambda(lambda x: (x ** jj)/math.factorial(jj), output_shape=x2Shape[1:])(x)]),temp])
    temp = Add()([Lambda(lambda x: x[:,:,0], output_shape=x2Shape[1:])(x2),temp])
    fc_1 =Lambda(lambda x: x*10000000., output_shape=x2Shape[1:])(Activation("tanh")(Lambda(lambda x: x/10000000., output_shape=x2Shape[1:])(temp)))
    fc_1 = Dense(256,trainable=True)(fc_1)   
    fc_1 = Dropout(0.2)(fc_1)
    feature = Dense(units=256, use_bias=True,trainable=True)(feature)
    feature = add([feature,fc_1])
    feature = Dropout(0.8)(feature)
    # feature = Dropout(0.6)(feature)
    # feature = GaussianDropout(0.8)(feature)

    # feature = AlphaDropout(0.6)(feature)

    fc_2 = Dense(units=256, activation='relu', use_bias=True,trainable=True)(feature)
    fc_2 = Dropout(0.2)(fc_2)
    fc_2 = Dense(units=256,trainable=True)(fc_2)
    feature = Dense(units=256, use_bias=True,trainable=True)(feature)
    feature = add([feature,fc_2])
    # attention = Dense(units=256, activation='sigmoid')(feature)
    # x_a = multiply([feature,attention])
    feature = Dropout(0.8)(feature)
    # feature = Dropout(0.4)(feature)
    # feature = GaussianDropout(0.7)(feature)

    # feature = AlphaDropout(0.6)(feature)
    fc_2 = Dense(units=128, activation='relu', use_bias=True)(feature)
    fc_2 = Dropout(0.2)(fc_2)
    fc_2 = Dense(units=128)(fc_2)
    feature = Dense(units=128, use_bias=True)(feature)
    feature = add([feature,fc_2])
    # feature = Dropout(0.2)(feature)

    # feature = GaussianDropout(0.2)(feature)
	
	
	
    # fc_4 = Dense(units=60, use_bias=True)(feature)
    # fc_4 = Activation('softmax')(fc_4)
    # network = Model(input=[up_0, up_1, down_0, down_1, down_02, down_12, down_2, down_3], outputs=fc_4)
    # network.load_weights(weight_path)
    fc_4 = Dense(units=55, use_bias=True)(feature)
    fc_4 = Activation('softmax')(fc_4)
    network = Model(input=[up_0, up_1, down_0, down_1, down_02, down_12, down_2, down_3], outputs=fc_4)
    return network