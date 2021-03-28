
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

def class10(path,className=10):
	files = os.listdir(path)
	imgAlea=[]
	i=0
	for im in files:
		
		nvIm=Image.open(path+chr(47)+im).convert('L')
		nvIm= resizeimage.resize_cover(nvIm, [28, 28])
		if(i==0):
			print(np.array(nvIm).shape,len(nvIm),nvIm[0,0])
		i=i+1
		imgAlea.append((np.array(nvIm).reshape(28*28),className))
	return imgAlea

def getDataPlus(test,train,imgAleatrain,imgAleaTest,isRandom=False):
	"""ajout de la classe d images quelconques"""
	data_test=np.vstack((test,imgAleaTest))
	data_train=np.vstack((train,imgAleatrain))
	return getDataMNIST(data_test,data_train)

"""
start=time.time()
dtest,dtrain=loadDataMNIST()
xtest,ytest,xtrain,ytrain=getDataMNIST(dtest,dtrain)
dataAlea=class10("ft")
xtest10,ytest10,xtrain10,ytrain10=getDataPlus(dtest,dtrain,dataAlea[:900],dataAlea[900:])
end=time.time()
print("tps : ",end-start )
"""
