import Q_learning as q
import pybullet as p
import cubesEnv as env
from torchvision.datasets import ImageFolder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import time
import create_file as cf
import creer_Env as ce
import toolbox


trans = transforms.Compose([transforms.ToTensor()])

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5,0.5), (0.5, 0.5, 0.5,0.5))])

net=q.Net()
optimizer = optim.SGD(net.parameters(), lr=0.01)
criterion = nn.MSELoss()
envi=ce.start_env()
ce.test_cam()
done=False

transform2=transforms.Compose([transforms.Grayscale(1),transforms.ToTensor(), transforms.Normalize((0.1307,),   (0.3081,))])

transform3=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, ), (0.5, ))])

while (not done):
    print("fonctionne ?")
    action = []
  

    p.stepSimulation()
    #state, reward, done, info = CubesEnv.step(action)
    obs = envi.getExtendedObservation()
    test=transform3(obs)
    test=test.unsqueeze(0)
    print(np.shape(test))
    res=net(test)
    print(res)
    ar=res.detach().numpy()
    print(ar)
    u=np.mean(ar,axis=0)
    print(np.shape(u))
    print(u)
