import keras
import random
from keras.layers import Input, Dense, Permute, Reshape
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D, GlobalMaxPooling2D
from keras.layers import Activation, Dropout, Flatten, concatenate, Maximum, Lambda, Multiply, Add, add, multiply, SpatialDropout2D, GaussianDropout, AlphaDropout
from keras.optimizers import Adam, Adadelta, RMSprop
from keras.regularizers import l2
from keras import backend as K
from keras.utils.training_utils import multi_gpu_model
from keras.engine.base_layer import Layer
from keras.legacy import interfaces
from keras.engine.base_layer import InputSpec
import tensorflow as tf
from tensorflow.python.ops import math_ops

epochs = 301
frames = 100   #64
frames_origin = 100  #64
batch_size_origin = 64 * 100 / frames_origin 
LayerThresh = 0.8
learning_rate = 0.00001 #0.00001
d_k = 64
input_shape = (frames, 49, 2)
input_shape3 = (frames-1, 49, 2)
input_shape4 = (frames-3, 49, 2)

input_shape2 = (frames, 48, 3)
largeBatch = 6400
use_bias = True
weight_path = 'TestBed_OpenPose_v4_COCO_6_9100.h5'  
save_path = 'Weight_save.h5'   
save_path2 = 'Weight_save_temp.h5'
long_term = 5
augmentation = False   
augmentation2 = True   
augmentation3 = True
augmentation_factor = 10
multi=False
if multi == True :
	batch_size_origin = batch_size_origin*2
from keras.utils.training_utils import multi_gpu_model

import math
taylor=3

def Rot(skeleton):   #(frames, 49, 3)
	newSkel = np.zeros_like(skeleton)
	for ii in range(skeleton.shape[0]):
		cos1 = 	1.#-random.random()/20.
		cos2 = 	1.-random.random()/2.
		# cos3 = 	1.#-random.random()/20.
		sin1 = (-1.)**(random.randint(1,2))*np.sqrt(1.-cos1**2)
		sin2 = (-1.)**(random.randint(1,2))*np.sqrt(1.-cos2**2)
		# sin3 = (-1.)**(random.randint(1,2))*np.sqrt(1.-cos3**2)
		newSkel[ii,:,:,0] = skeleton[ii,:,:,0]*(cos2*cos3)  + skeleton[ii,:,:,1]*(-cos2*sin3) 
		newSkel[ii,:,:,1] = skeleton[ii,:,:,0]*(cos1*sin3+sin1*sin2*cos3)  + skeleton[ii,:,:,1]*(cos1*cos3-sin1*sin2*sin3)
		# newSkel[ii,:,:,2] = skeleton[ii,:,:,0]*(sin1*sin3-cos1*sin2*cos3)  + skeleton[ii,:,:,1]*(sin1*cos3+cos1*sin2*sin3)   + skeleton[ii,:,:,2]*(cos1*cos2)


	return newSkel	
	
	
def Rot2D(skeleton):   #(frames, 49, 3)
	newSkel = np.zeros_like(skeleton)
	for ii in range(skeleton.shape[0]):
		cos2 = 	1.-random.random()/2.
		# cos3 = 	1.#-random.random()/20.
		sin2 = (-1.)**(random.randint(1,2))*np.sqrt(1.-cos2**2)
		# sin3 = (-1.)**(random.randint(1,2))*np.sqrt(1.-cos3**2)
		newSkel[ii,:,:,0] = skeleton[ii,:,:,0]*(cos2)  + skeleton[ii,:,:,1]*(sin2) 
		newSkel[ii,:,:,1] = skeleton[ii,:,:,0]*(-sin2)  + skeleton[ii,:,:,1]*(cos2)
		# newSkel[ii,:,:,2] = skeleton[ii,:,:,0]*(sin1*sin3-cos1*sin2*cos3)  + skeleton[ii,:,:,1]*(sin1*cos3+cos1*sin2*sin3)   + skeleton[ii,:,:,2]*(cos1*cos2)


	return newSkel	

Trainable = True
def share_stream(x_shape,outputNum=256):
    input = Input(x_shape)
    x_a=input	

    conv1 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1),  padding='same',
                   use_bias=use_bias, trainable=Trainable)(x_a)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1),  padding='same',
                   use_bias=use_bias, trainable=Trainable)(conv1)
    x_a = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1),  padding='same',
                   use_bias=use_bias,trainable=Trainable)(x_a)
    x_a = add([x_a,conv1])
    attention = Conv2D(filters=32, kernel_size=(5, 5), activation = 'sigmoid', strides=(1, 1),  padding='same',
                   use_bias=use_bias,trainable=Trainable)(x_a)
    x_a = multiply([x_a,attention])
    x_a = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x_a)
	
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),  padding='same',
                   use_bias=use_bias, trainable=Trainable)(x_a)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1),  padding='same',
                   use_bias=use_bias, trainable=Trainable)(conv1)
    x_a = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1),  padding='same',
                   use_bias=use_bias, trainable=Trainable)(x_a)
    x_a = add([x_a,conv1])

    x_a = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x_a)


    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same',
                   use_bias=use_bias, trainable=Trainable)(x_a)				   
    temp = Multiply()([Conv2D(filters = 128, kernel_size=(3, 3),padding = 'same', trainable=Trainable)(Lambda(lambda x: x*10000000.)(Activation("tanh")(Lambda(lambda x: x/10000000.)(x)))),x])
    for jj in range(2,taylor):
        temp = Add()([Multiply()([Conv2D(filters = 128, kernel_size=(3, 3),padding = 'same', trainable=Trainable)(Lambda(lambda x: x*10000000.)(Activation("tanh")(Lambda(lambda x: x/10000000.)(x)))),Lambda(lambda x: (x ** jj)/math.factorial(jj))(x)]),temp])
    temp = Add()([Conv2D(filters = 128, kernel_size=(3, 3),padding = 'same', trainable=Trainable)(Lambda(lambda x: x*10000000.)(Activation("tanh")(Lambda(lambda x: x/10000000.)(x)))),temp])
    conv2 =Lambda(lambda x: x*10000000.)(Activation("tanh")(Lambda(lambda x: x/10000000.)(temp)))
    conv2 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1),  padding='same',
                   use_bias=use_bias, trainable=Trainable)(conv2)
    x_a = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1),  padding='same',
                   use_bias=use_bias, trainable=Trainable)(x_a)
    x_a = add([x_a,conv2])
    x_a = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x_a)


    conv3 = Conv2D(filters=outputNum, kernel_size=(3, 3), strides=(1, 1), padding='same',
                   use_bias=use_bias, trainable=Trainable)(x_a)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(filters=outputNum, kernel_size=(1, 1),strides=(1, 1), padding='same',
                   use_bias=use_bias, trainable=Trainable)(conv3)
    x_a = Conv2D(filters=outputNum, kernel_size=(1, 1),strides=(1, 1),  padding='same',
                   use_bias=use_bias, trainable=Trainable)(x_a)
    x_a = add([x_a,conv3])

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

    feature = concatenate([up_feature, down_feature, down_feature2, down_feature3])

    x = Dense(units=256, use_bias=True, trainable=Trainable)(feature)
    x2 = Dense(16, trainable=Trainable)(Lambda(lambda x: x*10000000.)(Activation("tanh")(Lambda(lambda x: x/10000000.)(x))))
    x2Shape2=x2._keras_shape
    temp = Multiply()([Dense(256, trainable=Trainable)(Lambda(lambda x: x*10000000., output_shape=x2Shape2[1:])(Activation("tanh")(Lambda(lambda x: x/10000000., output_shape=x2Shape2[1:])(x2)))),x])
    for jj in range(2,taylor):
        temp = Add()([Multiply()([Dense(256, trainable=Trainable)(Lambda(lambda x: x*10000000., output_shape=x2Shape2[1:])(Activation("tanh")(Lambda(lambda x: x/10000000., output_shape=x2Shape2[1:])(x2)))),Lambda(lambda x: (x ** jj)/math.factorial(jj))(x)]),temp])
    temp = Add()([Dense(256, trainable=Trainable)(Lambda(lambda x: x*10000000., output_shape=x2Shape2[1:])(Activation("tanh")(Lambda(lambda x: x/10000000., output_shape=x2Shape2[1:])(x2)))),temp])
    fc_1 =Lambda(lambda x: x*10000000.)(Activation("tanh")(Lambda(lambda x: x/10000000.)(temp)))
    fc_1 = Dropout(0.4)(fc_1)

    fc_1 = Dense(256, trainable=True)(fc_1)   
    feature = Dense(units=256, use_bias=True, trainable=True)(feature)
    feature = add([feature,fc_1])
    feature = Dropout(0.8)(feature)


    fc_2 = Dense(units=128, activation='relu', use_bias=True,  trainable=True)(feature)
    fc_2 = Dropout(0.4)(fc_2)

    fc_2 = Dense(units=128,  trainable=Trainable)(fc_2)
    feature = Dense(units=128, use_bias=True,  trainable=True)(feature)
    feature = add([feature,fc_2])

    feature = Dropout(0.3)(feature)

    fc_4 = Dense(units=55, use_bias=True,)(feature)
    fc_4 = Activation('softmax')(fc_4)
    network = Model(input=[up_0, up_1, down_0, down_1, down_02, down_12, down_2, down_3], outputs=fc_4)

    return network

import numpy as np

def train_model():
	input_shape = (frames_origin, 31, 2)
	input_shape3 = (frames_origin-1, 31, 2)
	input_shape4 = (frames_origin-3, 31, 2)
	input_shape2 = (frames_origin, 30, 2)
	network = model((None,None,2),(None,None,2))
	global learning_rate
	adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
	network.summary()
	network.load_weights(weight_path)
	network.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
	network.save_weights(save_path2)

	
	batch_num = 0
	model_save_acc = 0
	all_train_accuracy = []
	all_train_loss = []
	all_tst_accuracy = []
	frames = frames_origin
	zero_metric = np.zeros([largeBatch,1, 31, 2], dtype=float)
	import scipy.io as sio
	import os
	Dir = os.listdir('./ETRI-Activity3D_Mat')
	train_list = []
	test_list = []

	for x in Dir:
		if int(x[6:9])%3==0:
			test_list.append(x)
		else:
			train_list.append(x)
	print(len(train_list))
	print(len(test_list))	
	
	
	tst_up_0 = np.zeros((len(test_list),frames, 18, 2))
	tst_up_1 = np.zeros((len(test_list),frames, 18, 2))
	tst_labels_0 = np.zeros((len(test_list),55))
	tstCnt = 0
	for num in range(len(test_list)):
		try:
			ftr = sio.loadmat(os.path.join('./ETRI-Activity3D_Mat', test_list[num]), verify_compressed_data_integrity=False)
			data = ftr['skeleton']
			if data.shape[0]>1:
				if data.shape[1]>frames:		
					ind = random.sample(range(0,data.shape[1]),frames) 
					ind.sort()
					data2 = data[:,ind,:,:]
					tst_up_0[tstCnt,:,:,:] = data2[0:1,:,[0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18],0:2]
					tst_up_1[tstCnt,:,:,:] = data2[1:2,:,[0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18],0:2]
				else :
					ind = [0]*frames
					ind[0:data.shape[1]] = range(data.shape[1])
					ind.sort()
					data2 = data[:,ind,:,:]
					tst_up_0[tstCnt,:,:,:] = data2[0:1,:,[0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18],0:2]
					tst_up_1[tstCnt,:,:,:] = data2[1:2,:,[0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18],0:2]                
			elif data.shape[0]==1:
				if data.shape[1]>frames:		
					ind = random.sample(range(0,data.shape[1]),frames) 
					ind.sort()
					data2 = data[:,ind,:,:]
					tst_up_0[tstCnt,:,:,:] = data2[:,:,[0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18],0:2]
				else :
					ind = [0]*frames
					ind[0:data.shape[1]] = range(data.shape[1])
					ind.sort()
					data2 = data[:,ind,:,:]
					tst_up_0[tstCnt,:,:,:] = data2[:,:,[0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18],0:2]
			tst_labels_0[tstCnt,int(test_list[num][1:4])-1] = 1
			tstCnt = tstCnt+1
		except:
			print(test_list[num])
	tst_labels_0 = tst_labels_0[:tstCnt,:]
	tst_up_0 = tst_up_0[:tstCnt,:,:,:]	
	tst_up_1 = tst_up_1[:tstCnt,:,:,:]		
	
	tst_up_0[:,:,:,0:1] = tst_up_0[:,:,:,0:1]-np.tile(tst_up_0[:,int(frames/2)-1:int(frames/2),0:1,0:1],(1,frames,18,1))
	tst_up_0[:,:,:,1:2] = tst_up_0[:,:,:,1:2]-np.tile(tst_up_0[:,int(frames/2)-1:int(frames/2),0:1,1:2],(1,frames,18,1))
	tst_up_1[:,:,:,0:1] = tst_up_1[:,:,:,0:1]-np.tile(tst_up_0[:,int(frames/2)-1:int(frames/2),0:1,0:1],(1,frames,18,1))
	tst_up_1[:,:,:,1:2] = tst_up_1[:,:,:,1:2]-np.tile(tst_up_0[:,int(frames/2)-1:int(frames/2),0:1,1:2],(1,frames,18,1))
	dist0 = np.sqrt((tst_up_0[:,int(frames/2)-1:int(frames/2),0:1,0:1]-tst_up_0[:,int(frames/2)-1:int(frames/2),1:2,0:1])**2+(tst_up_0[:,int(frames/2)-1:int(frames/2),0:1,1:2]-tst_up_0[:,int(frames/2)-1:int(frames/2),1:2,1:2])**2)
	dist0 = dist0+np.sqrt((tst_up_0[:,int(frames/2)-1:int(frames/2),1:2,0:1]-tst_up_0[:,int(frames/2)-1:int(frames/2),2:3,0:1])**2+(tst_up_0[:,int(frames/2)-1:int(frames/2),1:2,1:2]-tst_up_0[:,int(frames/2)-1:int(frames/2),2:3,1:2])**2)
	dist0 = dist0+np.sqrt((tst_up_0[:,int(frames/2)-1:int(frames/2),2:3,0:1]-tst_up_0[:,int(frames/2)-1:int(frames/2),3:4,0:1])**2+(tst_up_0[:,int(frames/2)-1:int(frames/2),2:3,1:2]-tst_up_0[:,int(frames/2)-1:int(frames/2),3:4,1:2])**2)

	tst_up_0 = tst_up_0/(np.tile(dist0,(1,frames, 18, 2))+0.0001)
	tst_up_1 = tst_up_1/(np.tile(dist0,(1,frames, 18, 2))+0.0001)
	tst_up_0 = tst_up_0[:,:,[16,14,0,1,2,3,4,3,2,1,8, 9,10,9,8,11,12,13,13,12,11,1,5,6,7,6,5,1,0,15,17],:]			
	tst_up_1 = tst_up_1[:,:,[16,14,0,1,2,3,4,3,2,1,8, 9,10,9,8,11,12,13,13,12,11,1,5,6,7,6,5,1,0,15,17],:]
	tst_down_0 = np.diff(tst_up_0,n=1,axis=1)
	tst_down_1 = np.diff(tst_up_1,n=1,axis=1)
	zero_metric2 = np.zeros([tst_up_0.shape[0],1, 31, 2], dtype=float)
	tst_down_0 = np.array(np.append(tst_down_0, zero_metric2, axis=1), dtype='float32')
	tst_down_1 = np.array(np.append(tst_down_1, zero_metric2, axis=1), dtype='float32')
	tst_down_02 = np.diff(tst_up_0,n=long_term,axis=1)
	tst_down_12 = np.diff(tst_up_1,n=long_term,axis=1)
	tst_down_02 = np.array(np.append(tst_down_02, np.tile(zero_metric2,(1,long_term,1,1)), axis=1), dtype='float32')
	tst_down_12 = np.array(np.append(tst_down_12, np.tile(zero_metric2,(1,long_term,1,1)), axis=1), dtype='float32')			
	tst_down_2 = np.diff(tst_up_0,n=1,axis=2)
	tst_down_3 = np.diff(tst_up_1,n=1,axis=2)	
	
	
				
	tst_loss = network.evaluate([tst_up_0, tst_up_1, tst_down_0, tst_down_1,tst_down_02,tst_down_12, tst_down_2, tst_down_3], tst_labels_0,batch_size=int(batch_size_origin))
	print(tst_loss)
	print('The test data accuracy: %r' % tst_loss[1])
	K.clear_session()

if __name__ == '__main__':

	train_model()
