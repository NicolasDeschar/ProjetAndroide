import loadData as *
import numpy as np

def learnBernoulli ( X,Y ):
    # votre code
    lp=np.zeros((10,784))
    for i in range(10):
        lp[i]=X[Y==i].sum(0)/X[Y==i].shape[0]
    return lp

def logpobsBernoulli(X, theta):
    # votre code ici
    res=np.zeros((10))
    theta=np.maximum(theta,1e-4)
    for i in range(10):
        for j in range(784):
            if(theta[i,j]>(1-1e-4) and (theta[i,j])<(1+1e-4)):
                theta[i,j]=1-1e-4

    for j in range(10): 
        res[j]=np.sum(X*np.log(theta[j])+(1-X)*np.log(1-theta[j]))
    return res

dtest,dtrain=loadDataMNIST()
xtest,ytest,xtrain,ytrain=getDataMNIST(dtest,dtrain)
dataAlea=class10("ft")
xtest10,ytest10,xtrain10,ytrain10=getDataPlus(dtest,dtrain,dataAlea[:900],dataAlea[900:])

Xb_train = np.where(xtrain>0, 1, 0)
Xb_test  = np.where(xtest>0, 1, 0)
theta = learnBernoulli ( Xb_train,ytrain )

 
#evaluation
def matrice_confusion(Y, Y_hat):
    # votre code
    C=np.zeros((10,10))
    for i in range(Y.shape[0]):
        #C[i,j]=Y[Y==i].shape[0]+Y_hat[Y_hat==j].shape[0]
        C[int(Y[i]),int(Y_hat[i])]+=1
    return C

Y_train_hat = [np.argmax(logpobsBernoulli(Xb_train[i], theta)) for i in range (len(Xb_train))]
m = matrice_confusion(ytrain, Y_train_hat)
print("Taux de bonne classification training: {}".format(np.where(ytrain == Y_train_hat, 1, 0).mean()))

Y_test_hat = [np.argmax(logpobsBernoulli(Xb_test[i], theta)) for i in range (len(Xb_test))]
m = matrice_confusion(ytest, Y_test_hat)
print("\nTaux de bonne classification testing: {}".format(np.where(ytest == Y_test_hat, 1, 0).mean()))
