import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import cubesEnv as env
import classifCNN as cl




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1=torch.nn.Conv2d(1,6,5)
        self.pool=torch.nn.MaxPool2d(2,2)
        self.conv2=torch.nn.Conv2d(6,14,5)
        self.fc1=torch.nn.Linear(14*4*4,100)
        self.fc3=torch.nn.Linear(100,12)

    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=x.view(-1,14*4*4)
        x=F.relu(self.fc1(x))
        x=self.fc3(x)
        return x


#fonction executant le nn sur la sequence
def calc_net(seq):
    imagesi,actionsi=seq
    output=map(net,imagesi)
    out_meaned=[]
    for i in range(len(output[0])):
        out_meaned.append(np.mean([output[j][i] for j in range(len(output))]))
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
class sequence:
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
        return (images[:i],actions[:i-1])



def deep_Q(nb_episodes=100,max_iter=5000,epsilon=0.01,alpha=0.1, learning_rate=0.01):
    #initialisation du NN
    actions=[0,1,2,3,4,5,6,7,8,9,10,11]
    net=Net()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    #creation de l'environnement
    env.start_env()
    #initialisation de l'historique
    history=[]
    sequences=[]
    for i in range(nb_episodes):
        #réinitialisation de l'environnement
        env.reset_env()
        #initialisation de la sequence
        sequences.append(sequence(env.get_image))
        for j in range(max_iter):
            #choix epsilon-greedy de l'action
            r=random.random()
            if r<epsilon:
                a=random.choice(actions)
            else:
                output=calc_net(s.get[j])
                a=np.argmax(output)
            #step
            state, reward, done, info=env.step(a)
            #update de la sequence
            img=env.get_image()
            sequences[i].update(img,a)
            #enregistrement de la transition dans l'historique
            reward=cl.calc_reward(img)
            history.append(transition(j,a,reward,j+1,i))
            #mini-batch
            sample=random.sample(history,1)
            sequence=sequences[sample.episode]
            #calcul de l'objectif
            if done :
                y_i=sample.reward
                break
            else :
                y_i=sample.reward+alpha*np.argmax(calc_net(sequence.get(sample.end)))

            #descente de gradient stochastique
            optimizer.zero_grad()
            output=calc_net(sequence.get(sample.start))
            loss = criterion(output[sample.action],y_i)
            loss.backward()
            optimizer.step()
    env.close()










        




