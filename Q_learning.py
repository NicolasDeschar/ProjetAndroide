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
import time



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1=torch.nn.Conv2d(4,8,5)
        self.pool=torch.nn.MaxPool2d(2,2)
        self.conv2=torch.nn.Conv2d(8,15,5)
        self.fc1=torch.nn.Linear(15*77*77,50)
        self.fc2=torch.nn.Linear(50,12)

    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=x.view(-1,15*77*77)
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        return x


#fonction executant le nn sur la sequence
def calc_net(seq,net):
    start=time.time()
    imagesi,actionsi=seq
    output_list=[]

    for i in range(len(imagesi)):
        transformed=tb.convert_from_image_to_tensor(imagesi[i])
        output=net(transformed)
        data=output.detach().numpy()
        output_list.append(data)
    if len(output_list)==1:
        while(len(output_list)==1):
            output_list=output_list[0]
        print("temps de calcul nn : ",time.time()-start)
        return output_list
    out_meaned=np.mean(output_list,axis=0)
    while(len(out_meaned)==1):
        out_meaned=out_meaned[0]
    print("temps de calcul nn : ",time.time()-start)
    return out_meaned


#structure de données stockant les transitions
class transition:
    def __init__(self,start,action,reward,end,episode,done):
        self.start=start
        self.action=action
        self.reward=reward
        self.end=end
        self.episode=episode
        self.done=done

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
        return (self.images[:i+1],self.actions[:i])



def deep_Q(nb_episodes=100,max_iter=5000,epsilon=0.5,alpha=0.1, learning_rate=0.01):
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
                output=calc_net(sequence.get(j),net)
                a=np.argmax(output)
            #step
            print("chosen action ", a)
            state, reward, done, info=envi.step(a,envi)
            print("reward",reward)
            #update de la sequence
            img=envi.getExtendedObservation()
            sequences[i].update(img,a)
            #enregistrement de la transition dans l'historique
            history.append(transition(j,a,reward,j+1,i,done))
            #mini-batch
            sample=random.sample(history,1)[0]
            sequence=sequences[sample.episode]
            #calcul de l'objectif
            if sample.done :
                y_i=sample.reward
                break
            else :
                y_i=sample.reward+alpha*np.max(calc_net(sequence.get(sample.end),net))

            #descente de gradient stochastique
            optimizer.zero_grad()
            output=calc_net(sequence.get(sample.start),net)
            t_output=torch.from_numpy(np.asarray(output))
            t_y_i=torch.from_numpy(np.asarray(y_i))
            t_output.requires_grad=True
            t_y_i.requires_grad=True


            loss = criterion(t_output.float(),t_y_i.float())
            loss.backward()
            optimizer.step()
            #sequences[i]=sequence
    env.close()










        




