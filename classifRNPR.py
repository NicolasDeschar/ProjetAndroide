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



#2 couches cachees
class RN(torch.nn.Module):
    def __init__(self,nbHiddenLayers,nbNeuronsPerLayer,n_in=28*28,n_out=11,): #3 couches cachees
        super().__init__()
        #fully connected layer
        self.dim_in = n_in
        self.dim_out = n_out
        self.n_hidden_layers=nbHiddenLayers
        self.nbNPerLayer=nbNeuronsPerLayer
        #self.fc=None
        if(self.n_hidden_layers > 0):
            self.fc1 = torch.nn.Linear(self.dim_in,self.nbNPerLayer,bias=True)# In -> first hidden
             # Hidden -> hidden
            self.fc2=torch.nn.Linear(self.nbNPerLayer,self.nbNPerLayer,bias=True)
            self.fc3=torch.nn.Linear(self.nbNPerLayer,self.dim_out,bias=True) # -> last hidden -> out
        else:
            self.fc1 = [torch.nn.Linear(self.dim_in,self.dim_out,bias=True)] # Single-layer perceptron
    
    def forward(self, x):
        fc=[self.fc1,self.fc2,self.fc3]
        #fc=[self.fc1,self.fc2,self.fc3,self.fc4]
        for i in range(self.n_hidden_layers):
            x=F.relu(fc[i](x))
        x=fc[-1](x)
        #return F.log_softmax(x,dim=1)
        return F.softmax(x,dim=1)
    
#fonction d'entrainement du classifieur
def train(nbEntr,NN,loader,lr=1e-3):
    start=time.time()
    optimizer=torch.optim.Adam(NN.parameters(),lr) #lr : learning rate
    lloss=[]
    for i in range(nbEntr):
        for x,y in loader:
            NN.zero_grad() #
            output=NN(x.view(-1,28*28))
            #loss=F.cross_entropy(output,y)
            loss=F.nll_loss(output,y)
            loss.backward()
            optimizer.step()
        lloss.append(loss.detach())
    end=time.time()
    #print("Time execution of ", nbEntr ," trainings of ",end-start," s" )
    return lloss

#test de performance du classifieur sur le dataset loader
def test(loader,NN):
    start=time.time()
    correct_class =torch.zeros((11))
    total=torch.zeros((11))
    correct=0
    with torch.no_grad():
        for x,y in loader:
            output=NN(x.view(-1,28*28))
            for i,img in enumerate(output):
                if(torch.argmax(img)==y[i]):
                    correct+=1
                    correct_class[y[i]]+=1
                total[y[i]]+=1
    end=time.time()
    print("Time execution :  ",end-start)
    return correct_class,total,correct/total.sum()

def odd(correct,total):
    """
    fonction pour connaitre le nombre de bonne solution de chaque classe
    correct : contient l
    """
    res= []
    for i in range(len(total)):
        res.append(correct[i]/total[i])
    return res

def testNbCouche(nbEntr=2):
    """
    renvoie les temps pour entrainement et test et les erreurs
    
    """
    ltimetrain=[]
    ltimetest=[]
    listloss=[]
    for i in range(10,100,10):
        print("numero : ",i)
        nbHiddenLayers = 2
        nbNeuronsPerLayer = i
        model=RN(nbHiddenLayers,nbNeuronsPerLayer)
        ss=time.time()
        tloss=train(nbEntr,model,train11_loader)
        ee=time.time()
        print("temps train ", ee-ss)
        print("loss final : ",np.mean(tloss))
        listloss.append(np.mean(tloss))
        ltimetrain.append(ee-ss)
        
        ss=time.time()
        c,t,s=test(test11_loader ,model)
        ee=time.time()
        ltimetest.append(ee-ss)
        print("temps test", ee-ss)
        print(s,c,t)
        #print(CNNodd(c,t),"\n _____________________________\n")
        print("\n _____________________________\n")
    return ltimetrain,ltimetest,listloss

def plot(savename,ltimetrain,ltimetest,listloss): #fonction pour tracer les courbes d erreurs
    """
    entree : 
    ltimetrain : temps pour entrainement
    ltimetest : temps pour test
    listloss : liste de la moyenne des erreurs
    """
    fig,axes=plt.subplots(3)
    fig.set_size_inches(20,15)
    plt.grid()
    xx=np.arange(10,100,10)
    axes[0].plot(xx,ltimetrain, label="Train time execution")
    axes[0].plot(xx,ltimetest, label="Test time execution")
    axes[0].legend(loc="upper right")
    axes[2].plot(xx,listloss, label="Train error")
    axes[1].plot(xx,ltimetest, label="Test time execution")
    axes[1].legend(loc="upper right")
    axes[2].legend(loc="upper right")
    axes[2].set_xlabel('number of neuron')
    axes[1].set_xlabel('number of neuron')
    axes[0].set_xlabel('number of neuron')
    axes[2].set_ylabel('error')
    axes[1].set_ylabel('time (s)')
    axes[0].set_ylabel('time (s)')
    plt.savefig(savename)
    plt.show()


if(__name__=="__main__"):
    root_dir="mnist_png"
    transform2=transforms.Compose([transforms.Grayscale(1),transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_data=ImageFolder(root=os.path.join(root_dir,"training"),transform=transform2)
    test_data=ImageFolder(root=os.path.join(root_dir,"testing"),transform=transform2)
    print(train_data.classes,test_data.classes)
    batch_size=50
    train11_loader=torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test11_loader =torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    nbHiddenLayers = 2
    nbNeuronsPerLayer = 5
    myNN=RN(nbHiddenLayers,nbNeuronsPerLayer)
    train(1,myNN,train11_loader,lr=1e-3)
    test(test11_loader,myNN)


