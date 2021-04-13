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
        self.fc1=torch.nn.Linear(14*4*4,100)
        #self.fc2=torch.nn.Linear(120,80)
        self.fc3=torch.nn.Linear(100,11)
    
    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=x.view(-1,14*4*4)
        x=F.relu(self.fc1(x))
        #x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x
        
        
crossEL=torch.nn.CrossEntropyLoss()
def CNNtrain(nbEntr,NN,loader,lr=1e-3):
    start=time.time()
    optimizer=torch.optim.Adam(NN.parameters(),lr) #lr : learning rate
    lloss=[]
    for i in range(nbEntr):
        for k,(x,y) in enumerate(loader):
            #NN.zero_grad() #
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
    res= []
    for i in range(len(total)):
        res.append(correct[i]/total[i])
        print(f'Accuracy of {i}:{res[i]}')
    return res




if (__name__ == "__main__"):
	root_dir="mnist_png"
	transform2=transforms.Compose([transforms.Grayscale(1),transforms.ToTensor(), transforms.Normalize((0.1307,), 	(0.3081,))])
	train_data=ImageFolder(root=os.path.join(root_dir,"training"),transform=transform2)
	test_data=ImageFolder(root=os.path.join(root_dir,"testing"),transform=transform2)
	print(train_data.classes,test_data.classes)
	batch_size=50
	train11_loader=torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
	test11_loader =torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
	cubes_data=ImageFolder(root="cubes/img",transform=transform2)
	batch_size=10 # tu as 10 images dans x quand tu feras for x,y in loader
	loader=torch.utils.data.DataLoader(cubes_data, batch_size=batch_size, shuffle=True)
	model=CNN()
	CNNtrain(2,model,loader)

	# test 

	ct,tt,st=CNNtest(loader,model)
	print(st)
	print(CNNodd(ct,tt))




