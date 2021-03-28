import numpy as np
import loadData as ld

dtest,dtrain=ld.loadDataMNIST()
xtest,ytest,xtrain,ytrain=ld.getDataMNIST(dtest,dtrain) #10 min for load data
## Suppress TF info messages

import os

def sigmoid(x):
    return 1./(1 + np.exp(-x))

def sigmoidDeriv(x):
    return x * (1 - x)

def softmax(x):
    y=np.exp(x)
    return y/np.sum(y)

def erreur(output_expected,output_layer):
    return output_expected-output_layer

def erreur_mc(expected, output):
    return ((expected-output)**2).mean()

class SimpleNeural():
    def __init__(self, n_in, n_out, n_hidden_layers=1, n_neurons_per_hidden=100, params=None):
        self.dim_in = n_in
        self.dim_out = n_out
        self.n_per_hidden = n_neurons_per_hidden
        self.n_hidden_layers = n_hidden_layers
        self.weights = None 
        self.n_weights = None
        self.init_random_params()
        self.out = np.zeros(n_out)
        #print("Creating a simple mlp with %d inputs, %d outputs, %d hidden layers and %d neurons per layer"%(n_in, n_out,n_hidden_layers, n_neurons_per_hidden))
        self.output_layer=None #hiden -> output
        self.input_layer=None #input
        self.hidden_layers=None #intput ->hidden ->hidden
        
    
    def init_random_params(self):
        if(self.n_hidden_layers > 0):
            self.weights = [np.random.random((self.dim_in,self.n_per_hidden))] # In -> first hidden
            self.bias = [np.random.random(self.n_per_hidden)] # In -> first hidden
            self.weights.append(np.random.random((self.n_per_hidden,self.dim_out))) # -> last hidden -> out
            self.bias.append(np.random.random(self.dim_out))
        else:
            self.weights = [np.random.random((self.dim_in,self.dim_out))] # Single-layer perceptron
            self.bias = [np.random.random(self.dim_out)]
        self.n_weights = np.sum([w.size for w in self.weights]) + np.sum([b.size for b in self.bias])

    def get_parameters(self):
        """
        Returns all network parameters as a single array
        """
        flat_weights = np.hstack([arr.flatten() for arr in (self.weights+self.bias)])
        return flat_weights
    
    def feedForward(self,input_layer):
        self.input_layer=input_layer
        if(self.n_hidden_layers > 0):
            #Input
            y = sigmoid(self.input_layer@self.weights[0]+ self.bias[0])
            self.hidden_layers=y
            # Out
            a = y@self.weights[-1] + self.bias[-1]
            out = softmax(a)
            self.output_layer=out
            return out
        else :
            self.output_layer=softmax(x@self.weights[0]+ self.bias[0])
            return self.output_layer
        
    def predict(self,input_layer):
        self.input_layer=input_layer
        if(self.n_hidden_layers > 0):
            #Input
            y = sigmoid(self.input_layer@self.weights[0])
            self.hidden_layers=y
            # Out
            a = y@self.weights[-1]
            out = softmax(a)
            self.output_layer=out
            return out
        else :
            self.output_layer=softmax(x@self.weights[0])
            return np.argmax(self.output_layer)
        
        
    def backpropagation(self,output_expected):
        #calcul de l erreur
        output_layer_error=erreur(output_expected,self.output_layer)
        output_layer_delta=output_layer_error*sigmoidDeriv(self.output_layer)
        hl_error=np.dot(output_layer_delta.reshape((1,output_layer_delta.shape[0])),self.weights[-1].T)
        hl_delta=hl_error*sigmoidDeriv(self.hidden_layers)
        #correction
        a=np.reshape(self.input_layer,(self.input_layer.shape[0],1))
        self.weights[0]+=a@hl_delta
        a=np.reshape(self.hidden_layers,(self.hidden_layers.shape[0],1))
        b=np.reshape(output_layer_delta,(output_layer_delta.shape[0],1))
        self.weights[-1]+=a@b.T
        return np.mean(np.abs(output_layer_error))

        
    def set_parameters(self, flat_parameters):
        """
        Set all network parameters from a single array
        """
        i = 0 # index
        to_set = []
        self.weights = list()
        self.bias = list()
        if(self.n_hidden_layers > 0):
            # In -> first hidden
            w0 = np.array(flat_parameters[i:(i+self.dim_in*self.n_per_hidden)])
            self.weights.append(w0.reshape(self.dim_in,self.n_per_hidden))
            i += self.dim_in*self.n_per_hidden
            for l in range(self.n_hidden_layers-1): # Hidden -> hidden
                w = np.array(flat_parameters[i:(i+self.n_per_hidden*self.n_per_hidden)])
                self.weights.append(w.reshape((self.n_per_hidden,self.n_per_hidden)))
                i += self.n_per_hidden*self.n_per_hidden
            # -> last hidden -> out
            wN = np.array(flat_parameters[i:(i+self.n_per_hidden*self.dim_out)])
            self.weights.append(wN.reshape((self.n_per_hidden,self.dim_out)))
            i += self.n_per_hidden*self.dim_out
            # Samefor bias now
            # In -> first hidden
            b0 = np.array(flat_parameters[i:(i+self.n_per_hidden)])
            self.bias.append(b0)
            i += self.n_per_hidden
            for l in range(self.n_hidden_layers-1): # Hidden -> hidden
                b = np.array(flat_parameters[i:(i+self.n_per_hidden)])
                self.bias.append(b)
                i += self.n_per_hidden
            # -> last hidden -> out
            bN = np.array(flat_parameters[i:(i+self.dim_out)])
            self.bias.append(bN)
            i += self.dim_out
        else:
            n_w = self.dim_in*self.dim_out
            w = np.array(flat_parameters[:n_w])
            self.weights = [w.reshape((self.dim_in,self.dim_out))]
            self.bias = [np.array(flat_parameters[n_w:])]
        self.n_weights = np.sum([np.product(w.shape) for w in self.weights]) + np.sum([np.product(b.shape) for b in self.bias])
    
    
nb_it_training=1000
nbInputs = 784
nbOutputs = 10
nbHiddenLayers = 1
nbNeuronsPerLayer = 10
nn=SimpleNeural(nbInputs,nbOutputs,nbHiddenLayers,nbNeuronsPerLayer) 

# running NN
i=0
seuil=0.1
_error=10
while(i<nb_it_training and _error>seuil):
    observation = xtrain[9999]
    outputValues = nn.feedForward(observation)
    _error=nn.backpropagation(np.eye(10)[np.array(ytrain[9999])])
    #print ("Output values:", outputValues)
outputValues = np.argmax(nn.feedForward(observation))
print ("Output values:", outputValues,"expected ", ytrain[9999])