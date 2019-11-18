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
from keras.utils.training_utils import multi_gpu_model
import numpy as np
import csv
from ActionModel import model as ActModel

epochs = 301
frames = 64   #64
frames_origin = 64  #64
batch_size_origin = 64 * 100 / frames_origin 

learning_rate = 0.00001 #0.00001
d_k = 64
input_shape = (frames, 49, 3)
input_shape3 = (frames-1, 49, 3)
input_shape4 = (frames-3, 49, 3)
0
input_shape2 = (frames, 48, 3)
largeBatch = 6400
weight_path = './weights/Trained_Weights.h5'  
long_term = 5




def dataRead(filename, frames):
	 
	# f = open('D:\Database\Action_Recognition\TestBed\JointCSV(P001-P050)\A048_P013_G001_C005.csv', 'r', encoding='utf-8')
	f = open(filename, 'r', encoding='utf-8')
	Label = np.zeros((55))
	Label[int(filename[-21:-19])-1]=1
	# print(filename)
	rdr = csv.reader(f)
	whole = []
	for line in rdr:
		whole.append(line)
		
	f.close()    

	Data1 = np.zeros((max(1,len(whole)), 25, 3))
	Data2 = np.zeros((max(1,len(whole)), 25, 3))
	Occupied = np.zeros((max(1,len(whole))))
	cnt = -1
	tmp = [1]
	if len(whole)==0:
		print(filename)
	for ii in range(1,len(whole)):
		tmp = np.array(whole[ii],dtype='float32')
		# print(Occupied[cnt])
		if 	len(tmp)!=1:
			if Occupied[int(tmp[0])-1] == 0:
				Data1[int(tmp[0])-1,:,0] = tmp[3::10]
				Data1[int(tmp[0])-1,:,1] = tmp[4::10]
				Data1[int(tmp[0])-1,:,2] = tmp[5::10]
				Occupied[int(tmp[0])-1] = 1
			else :
				Data2[int(tmp[0])-1,:,0] = tmp[3::10]
				Data2[int(tmp[0])-1,:,1] = tmp[4::10]
				Data2[int(tmp[0])-1,:,2] = tmp[5::10]	

	Data1 = Data1[0:int(tmp[0]),:,:]
	Data2 = Data2[0:int(tmp[0]),:,:]	
	if Data1.shape[0]>frames:		
		ind = random.sample(range(0,Data1.shape[0]),frames) 
		ind.sort()
		Data1 = Data1[ind,:,0:3]
		Data2 = Data2[ind,:,0:3]
	else :
		ind = [0]*frames
		ind[0:Data1.shape[0]] = range(Data1.shape[0])
		ind.sort()
		Data1 = Data1[ind,:,0:3]
		Data2 = Data2[ind,:,0:3]
		
	return Data1,Data2, Label	
	

def PreProcessing(tst_up_0, tst_up_1,long_term):

	tst_up_0[:,:,:,0:1] = tst_up_0[:,:,:,0:1]-np.tile(tst_up_0[:,int(frames/2)-1:int(frames/2),0:1,0:1],(1,frames,25,1))
	tst_up_0[:,:,:,1:2] = tst_up_0[:,:,:,1:2]-np.tile(tst_up_0[:,int(frames/2)-1:int(frames/2),0:1,1:2],(1,frames,25,1))
	tst_up_0[:,:,:,2:3] = tst_up_0[:,:,:,2:3]-np.tile(tst_up_0[:,int(frames/2)-1:int(frames/2),0:1,2:3],(1,frames,25,1))
	tst_up_1[:,:,:,0:1] = tst_up_1[:,:,:,0:1]-np.tile(tst_up_0[:,int(frames/2)-1:int(frames/2),0:1,0:1],(1,frames,25,1))
	tst_up_1[:,:,:,1:2] = tst_up_1[:,:,:,1:2]-np.tile(tst_up_0[:,int(frames/2)-1:int(frames/2),0:1,1:2],(1,frames,25,1))
	tst_up_1[:,:,:,2:3] = tst_up_1[:,:,:,2:3]-np.tile(tst_up_0[:,int(frames/2)-1:int(frames/2),0:1,2:3],(1,frames,25,1))				
	dist0 = np.sqrt((tst_up_0[:,int(frames/2)-1:int(frames/2),0:1,0:1]-tst_up_0[:,int(frames/2)-1:int(frames/2),1:2,0:1])**2+(tst_up_0[:,int(frames/2)-1:int(frames/2),0:1,1:2]-tst_up_0[:,int(frames/2)-1:int(frames/2),1:2,1:2])**2+(tst_up_0[:,int(frames/2)-1:int(frames/2),0:1,2:3]-tst_up_0[:,int(frames/2)-1:int(frames/2),1:2,2:3])**2)
	dist0 = dist0+np.sqrt((tst_up_0[:,int(frames/2)-1:int(frames/2),1:2,0:1]-tst_up_0[:,int(frames/2)-1:int(frames/2),2:3,0:1])**2+(tst_up_0[:,int(frames/2)-1:int(frames/2),1:2,1:2]-tst_up_0[:,int(frames/2)-1:int(frames/2),2:3,1:2])**2+(tst_up_0[:,int(frames/2)-1:int(frames/2),1:2,2:3]-tst_up_0[:,int(frames/2)-1:int(frames/2),2:3,2:3])**2)
	dist0 = dist0+np.sqrt((tst_up_0[:,int(frames/2)-1:int(frames/2),2:3,0:1]-tst_up_0[:,int(frames/2)-1:int(frames/2),3:4,0:1])**2+(tst_up_0[:,int(frames/2)-1:int(frames/2),2:3,1:2]-tst_up_0[:,int(frames/2)-1:int(frames/2),3:4,1:2])**2+(tst_up_0[:,int(frames/2)-1:int(frames/2),2:3,2:3]-tst_up_0[:,int(frames/2)-1:int(frames/2),3:4,2:3])**2)

	tst_up_0 = tst_up_0/(np.tile(dist0,(1,frames, 25, 3))+0.0001)
	tst_up_1 = tst_up_1/(np.tile(dist0,(1,frames, 25, 3))+0.0001)
	tst_up_0 = tst_up_0[:,:,[3,2,20,4,5,6,7,21,7,22,7,6,5,4,20,1,0,12,13,14,15,14,13,12,0,16,17,18,19,18,17,16,0,1,20,8,9,10,11,23,11,24,11,10,9,8,20,2,3],:]			
	tst_up_1 = tst_up_1[:,:,[3,2,20,4,5,6,7,21,7,22,7,6,5,4,20,1,0,12,13,14,15,14,13,12,0,16,17,18,19,18,17,16,0,1,20,8,9,10,11,23,11,24,11,10,9,8,20,2,3],:]
	tst_down_0 = np.diff(tst_up_0,n=1,axis=1)
	tst_down_1 = np.diff(tst_up_1,n=1,axis=1)
	zero_metric2 = np.zeros([tst_up_0.shape[0],1, 49, 3], dtype=float)
	tst_down_0 = np.array(np.append(tst_down_0, zero_metric2, axis=1), dtype='float32')
	tst_down_1 = np.array(np.append(tst_down_1, zero_metric2, axis=1), dtype='float32')
	tst_down_02 = np.diff(tst_up_0,n=long_term,axis=1)
	tst_down_12 = np.diff(tst_up_1,n=long_term,axis=1)
	tst_down_02 = np.array(np.append(tst_down_02, np.tile(zero_metric2,(1,long_term,1,1)), axis=1), dtype='float32')
	tst_down_12 = np.array(np.append(tst_down_12, np.tile(zero_metric2,(1,long_term,1,1)), axis=1), dtype='float32')			
	tst_down_2 = np.diff(tst_up_0,n=1,axis=2)
	tst_down_3 = np.diff(tst_up_1,n=1,axis=2)	
	
	return tst_up_0,tst_up_1, tst_down_0, tst_down_1, tst_down_02, tst_down_12, tst_down_2, tst_down_3
	
	
def train_model():

	network = ActModel((frames, 49, 3),(frames, 48, 3))
	network.summary()
	network.load_weights(weight_path,by_name=True)
	# network.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


	tst_up_0 = np.zeros((1,frames, 25, 3))
	tst_up_1 = np.zeros((1,frames, 25, 3))
	tst_labels_0 = np.zeros((1,55))
	tst_up_0[0,:,:,:], tst_up_1[0,:,:,:], tst_labels_0[0,:] = dataRead('./samples/A001_P001_G001_C005.csv', frames) #tst_data.get_tst_single_data(tst_data_name[num], tst_cursors[num], 0)
	tst_up_0,tst_up_1, tst_down_0, tst_down_1, tst_down_02, tst_down_12, tst_down_2, tst_down_3 = PreProcessing(tst_up_0, tst_up_1,long_term = long_term)
	# tst_loss = network.evaluate([tst_up_0, tst_up_1, tst_down_0, tst_down_1,tst_down_02,tst_down_12, tst_down_2, tst_down_3], tst_labels_0)
	# print(tst_loss)
	output = network.predict([tst_up_0, tst_up_1, tst_down_0, tst_down_1,tst_down_02,tst_down_12, tst_down_2, tst_down_3])
	print('Predicted Action Class : ', np.argmax(output) + 1)

if __name__ == '__main__':
	with K.tf.device('/gpu:1'):
		# network = model()
		train_model()
