import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.autograd import Function

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pylab import *
from resizeimage import resizeimage
import os
import time


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=torch.nn.Conv2d(1,6,5)
        self.pool=torch.nn.MaxPool2d(2,2)
        self.conv2=torch.nn.Conv2d(6,14,5)
        self.fc1=torch.nn.Linear(14*4*4,30)
        self.fc2=torch.nn.Linear(30,40)
        self.fc3=torch.nn.Linear(40,11)
    
    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=x.view(-1,14*4*4)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.softmax(x,dim=1)
        return x
        
        
crossEL=torch.nn.CrossEntropyLoss()
def CNNtrain(nbEntr,NN,loader,lr=1e-3):
    start=time.time()
    optimizer=torch.optim.Adam(NN.parameters(),lr) #lr : learning rate
    lloss=[]
    for i in range(nbEntr):
        w=0
        for k,(x,y) in enumerate(loader):
            if w%100==0:
                print("done ",w,"of ",len(loader))
            w+=1
            NN.zero_grad() #
            output=NN(x)
            loss=crossEL(output,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        lloss.append(loss)
        if(k+1)%1000==0:
            print(f'Loss : {loss.item(): 4f}')
    end=time.time()
    print("Time execution of ", nbEntr ," trainings of ",end-start," s" )
    return lloss


def CNNtest(loader,NN):
    start=time.time()
    correct_class =torch.zeros((11))
    total=torch.zeros((11))
    correct=0
    with torch.no_grad():
        for x,y in loader:
            output=NN(x) #x contient 50 images
            for i,img in enumerate(output): #ici je parcours les images dans le output il y 50 resultats
                print(i,img,torch.argmax(img),y[i]) 
                print(img.numpy())
                if(torch.argmax(img)==y[i]):
                    correct+=1
                    correct_class[y[i]]+=1
                    plt.imshow(x[i].view(28,28))
                    plt.title("Predicted class {}".format(torch.argmax(img)))
                    plt.show()
                total[y[i]]+=1
                #plt.imshow(x[i].view(28,28))
                #plt.title("Predicted class {}".format(torch.argmax(img)))
                #plt.show()
        
    end=time.time()
    print("Time execution : ",end -start)
    return correct_class,total,correct/total.sum()

def CNNodd(correct,total):
    """pour connaitre la proba de reussite dans chacun des classes
	correct : correst_class de la sortie de CNNtest, 
	total : total de la sortie de CNNtest
    """
    res= []
    for i in range(len(total)):
        res.append(correct[i]/total[i])
        print(f'Accuracy of {i}:{res[i]}')
    return res


def create_ClassifDATA(_batch_size,_transform,_root,_shuffle):
    data=ImageFolder(root=_root,transform=_transform)
    loader=torch.utils.data.DataLoader(data, batch_size=_batch_size, shuffle=_shuffle)
    return data,loader

if (__name__ == "__main__"):
	transf=transforms.Compose([transforms.Grayscale(1),transforms.ToTensor(), transforms.Normalize((0.1307,), 	(0.3081,))])

	model=CNN()
	data,loader=create_ClassifDATA(50,transf,"mnist_png/training",True)
	CNNtrain(1,model,loader)


	ct,tt,st=CNNtest(loader,model)
	print(st)
	print(CNNodd(ct,tt))




