import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import cubesEnv as env
import classifCNN as cl
import creer_Env as ce
import toolbox as tb




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1=torch.nn.Conv2d(4,25,5)
        self.pool=torch.nn.MaxPool2d(2,2)
        self.conv2=torch.nn.Conv2d(25,100,5)
        self.fc1=torch.nn.Linear(100*4*4,10000)
        self.fc2=torch.nn.Linear(10000,1000)
        self.fc3=torch.nn.Linear(1000,12)

    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=x.view(-1,100*4*4)
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        x=self.fc3(x)
        return x


#fonction executant le nn sur la sequence
def calc_net(seq):
    imagesi,actionsi=seq
    output_flattend=[]

    for i in range(len(imagesi)):
        transformed=tb.convert_from_image_to_tensor(imagesi[i])
        output=net(transformed)
        output_flattend.append(np.mean(output,axis=0))
    out_meaned=np.mean(output_flattend,axis=0)
    return out_meaned


#structure de données stockant les transitions
class transition:
    def __init__(self,start,action,reward,end,episode):
        self.start=start
        self.action=action
        self.rewrad=reward
        self.end=end
        self.episode=episode

#structure de données stockant la séquence
class Sequence:
    def __init__(self,image):
        self.images=[]
        self.actions=[]
        self.images.append(image)

    #ajoute une image et une action a la séquence
    def update(self,image,action):
        self.images.append(image)
        self.actions.append(action)

    #renvoie la séquence au temps i
    def get(self,i):
        return (self.images[:i],self.actions[:i-1])



def deep_Q(nb_episodes=100,max_iter=5000,epsilon=0.01,alpha=0.1, learning_rate=0.01):
    #initialisation du NN
    actions=[0,1,2,3,4,5,6,7,8,9,10,11]
    net=Net()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    #creation de l'environnement
    envi=ce.start_env()
    #initialisation de l'historique
    history=[]
    sequences=[]
    for i in range(nb_episodes):
        #réinitialisation de l'environnement
        ce.reset_env(envi)
        #initialisation de la sequence
        sequence=Sequence(envi.getExtendedObservation())
        sequences.append(sequence)
        for j in range(max_iter):

            #choix epsilon-greedy de l'action
            r=random.random()
            if r<epsilon:
                a=random.choice(actions)
            else:
                output=calc_net(sequence.get(j))
                a=np.argmax(output)
            #step
            state, reward, done, info=envi.step(a)
            #update de la sequence
            img=envi.getExtendedObservation()
            sequences[i].update(img,a)
            #enregistrement de la transition dans l'historique
            history.append(transition(j,a,reward,j+1,i))
            #mini-batch
            sample=random.sample(history,1)
            sequence=sequences[sample.episode]
            #calcul de l'objectif
            if done :
                y_i=sample.reward
                break
            else :
                y_i=sample.reward+alpha*np.max(calc_net(sequence.get(sample.end)))

            #descente de gradient stochastique
            optimizer.zero_grad()
            output=calc_net(sequence.get(sample.start))
            t_output=torch.from_numpy(output)
            t_y_i=torch.from_numpy(y_i)
            loss = criterion(output[sample.action],y_i)
            loss.backward()
            optimizer.step()
            sequences[i]=sequence
    env.close()










        




