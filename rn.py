import pybullet as p
import numpy as np
import pybullet_data
import time
from PIL import Image
from pylab import *
from toolbox import discreteProb
import create_file

import shutil
import os

np.random.seed(1)

def loadDataMNIST(path="mnist_png"):
	"""path of mnist_png
	return list(tuple(array[int],int)) data_test,data_train"""
	data_train=[]
	data_test=[]
	for i in range(10):
		pathTest =path+chr(47)+"testing"+chr(47)+str(i)
		pathTrain=path+chr(47)+"training"+chr(47)+str(i)
		filesTest = os.listdir(pathTest)
		filesTrain = os.listdir(pathTrain)
		for name in filesTest:
			data_test.append((np.reshape(np.array(Image.open(pathTest+chr(47)+name)),28*28),i))
		for name in filesTrain:
			data_train.append((np.reshape(np.array(Image.open(pathTrain+chr(47)+name)),28*28),i))
	return data_test,data_train

def getDataMNIST(test,train,isRandom=False):
	"""return the image data : 
	x_test,y_test,x_train,y_train

	x_test shape : (number of img, 28*28) array(array)
	y_test shape : (number of img) array(int)

	path of mnist_png
	isRandom ==True modify the sequence
	"""
	#modify the sequence
	if(isRandom):
		np.random.shuffle(train)
		np.random.shuffle(test)
	x_train=[]
	y_train=[]
	x_test=[]
	y_test=[]
	for data in train:
		x_train.append(data[0])
		y_train.append(data[1])
	for data in test:
		x_test.append(data[0])
		y_test.append(data[1])	
	return np.array(x_test),np.array(y_test),np.array(x_train),np.array(y_train)


dtest,dtrain=loadDataMNIST()
xtest,ytest,xtrain,ytrain=getDataMNIST(dtest,dtrain)

