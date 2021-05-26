#fichier pour utilisation de la methode du maximum de vraisemblance
#Pour creer les donnees pour le classifMV.py

import numpy as np
import time
from PIL import Image
from pylab import *
import create_file
from resizeimage import resizeimage
import shutil 
import os

#np.random.seed(1)

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
	"""renvoie les donnees d images : 
	x_test,y_test,x_train,y_train

	x_test shape : (number of img, 28*28) array(array)
	y_test shape : (number of img) array(int)

	path of mnist_png
	isRandom ==True modifier l ordre des images dans liste d images
	"""
	#modifier la sequence
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

def class10(path,className=10):
	"""
	creer la sequence d images pour la 11e classe
	"""
	files = os.listdir(path)
	imgAlea=[]
	i=0
	for im in files:
		nvIm=Image.open(path+chr(47)+im).convert('L')
		nvIm= resizeimage.resize_cover(nvIm, [28, 28])
		imgAlea.append((np.array(nvIm).reshape(28*28),className))
	return np.array(imgAlea)

def getDataPlus(test,train,imgAleatrain,imgAleaTest,isRandom=False):
	"""
	combiner les listes d images 
	"""
	"""ajout de la classe d images quelconques"""
	data_test=np.vstack((test,imgAleaTest))
	data_train=np.vstack((train,imgAleatrain))
	return getDataMNIST(data_test,data_train)

if __name__=='__main__': #creation des donnees d images
	start=time.time()
	dtest,dtrain=loadDataMNIST()
	xtest,ytest,xtrain,ytrain=getDataMNIST(dtest,dtrain)
	dataAleatrain=class10("mnist_png/training/tout")
	dataAleatest=class10("mnist_png/testing/tout")
	xtest10,ytest10,xtrain10,ytrain10=getDataPlus(dtest,dtrain,dataAleatrain,dataAleatest)
	end=time.time()
	print("tps pour charger les donnees : ",end-start )

