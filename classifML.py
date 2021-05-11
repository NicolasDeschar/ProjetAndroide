import loadData as ld
import numpy as np
import time
from resizeimage import resizeimage
from PIL import Image
from pylab import *

def learnBernoulli ( X,Y,nbClass=10 ):
    lp=np.zeros((nbClass,X.shape[1]))
    for i in range(nbClass):
        lp[i]=X[Y==i].sum(0)/X[Y==i].shape[0]
    return lp

def logpobsBernoulli(X, theta,nbClass=10,confiance=1e-4):
    res=np.zeros((nbClass))
    theta=np.maximum(theta,confiance)
    for i in range(nbClass):
        for j in range(len(X)):
            if(theta[i,j]>(1-confiance) and (theta[i,j])<(1+1e-4)):
                theta[i,j]=1-confiance

    for j in range(nbClass): 
        res[j]=np.sum(X*np.log(theta[j])+(1-X)*np.log(1-theta[j]))
    return res
    
if __name__=='__main__':
    start=time.time()
    dtest,dtrain=ld.loadDataMNIST()
    xtest,ytest,xtrain,ytrain=ld.getDataMNIST(dtest,dtrain)
    dataAleatrain=ld.class10("mnist_png/training/tout")
    dataAleatest=ld.class10("mnist_png/testing/tout")
    xtest10,ytest10,xtrain10,ytrain10=ld.getDataPlus(dtest,dtrain,dataAleatrain,dataAleatest)
    end=time.time()
    print("Temps pour charger les donnees : ",end-start)
    start=time.time()
    Xb_train = np.where(xtrain>0, 1, 0)
    Xb_test  = np.where(xtest>0, 1, 0)
    theta = learnBernoulli ( Xb_train,ytrain )
    end=time.time()
    print("Temps pour trouver le parametre : ",end-start)
    #evaluation
    start=time.time()
    Y_train_hat = [np.argmax(logpobsBernoulli(Xb_train[i], theta)) for i in range (len(Xb_train))]
    end=time.time()
    print("Taux de bonne classification training: {}".format(np.where(ytrain == Y_train_hat, 1, 0).mean()))
    print("Temps pour classifier les donnees d'entrainement ",end-start)
    start=time.time()
    Y_test_hat = [np.argmax(logpobsBernoulli(Xb_test[i], theta)) for i in range (len(Xb_test))]
    end=time.time()
    print("Taux de bonne classification testing: {}".format(np.where(ytest == Y_test_hat, 1, 0).mean()))
    print("Temps pour classifier les donnees d'entrainement ",end-start)
    """
    

    
    start=time.time()
    mnist2=Image.open("ft/magazine-unlock-01-2.3.1152-_82D7E759A7CE59589C97850534E61A53.png").convert('L')
    cover=resizeimage.resize_cover(mnist2, [28, 28])
    cover=np.array(cover).reshape(28*28)
    print(cover.shape,cover[0],len(cover))
    print("L'image est de classe :", np.argmax(logpobsBernoulli(cover, theta)))
    end=time.time()
    print("temps ",end-start)
    """