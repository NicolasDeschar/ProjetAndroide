import Q_learning as q
import pybullet as p
import cubesEnv as env
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import time

trans = transforms.Compose([transforms.ToTensor()])

net=q.Net()
optimizer = optim.SGD(net.parameters(), lr=0.01)
criterion = nn.MSELoss()
envi=env.start_env()
env.test_cam()
done=False

transform2=transforms.Compose([transforms.Grayscale(1),transforms.ToTensor(), transforms.Normalize((0.1307,),   (0.3081,))])

pixelWidth = 320
pixelHeight = 220


while (not done):
    print("fonctionne ?")
    action = []
   

    p.stepSimulation()
    #state, reward, done, info = CubesEnv.step(action)
    obs = envi.getExtendedObservation()
    print(np.shape(obs))
    b=trans(obs)
    res=net(b)
    '''
    viewMatrix = [1.0, 0.0, -0.0, 0.0, -0.0, 0.1736481785774231, -0.9848078489303589, 0.0, 0.0, 0.9848078489303589, 0.1736481785774231, 0.0, -0.0, -5.960464477539063e-08, -4.0, 1.0]
    projectionMatrix = [
        1.0825318098068237, 0.0, 0.0, 0.0, 0.0, 1.732050895690918, 0.0, 0.0, 0.0, 0.0,
        -1.0002000331878662, -1.0, 0.0, 0.0, -0.020002000033855438, 0.0]
    img_arr = p.getCameraImage(pixelWidth,
                           pixelHeight,
                           viewMatrix=viewMatrix,
                           projectionMatrix=projectionMatrix,
                           shadow=1,
                           lightDirection=[1, 1, 1])
'''
'''
startPos = [0,0,1]
startOrientation = p.getQuaternionFromEuler([0,0,0])
boxId = p.loadURDF("r2d2.urdf",startPos, startOrientation)


camTargetPos = [0, 0, 0]
camDistance = 2
pitch = -15
upAxisIndex = 2
yaw=0
roll=10

while (p.isConnected()):
  for i in range(-100, 10000, 1):
    print(roll)
    start = time.time()
    p.stepSimulation()
    stop = time.time()
    print("stepSimulation %f" % (stop - start))

    #viewMatrix = [1.0, 0.0, -0.0, 0.0, -0.0, 0.1736481785774231, -0.9848078489303589, 0.0, 0.0, 0.9848078489303589, 0.1736481785774231, 0.0, -0.0, -5.960464477539063e-08, -4.0, 1.0]
    viewMatrix = p.computeViewMatrixFromYawPitchRoll([0,i,0], camDistance, yaw, pitch, roll,
                                                     upAxisIndex)
    projectionMatrix = [
        1.0825318098068237, 0.0, 0.0, 0.0, 0.0, 1.732050895690918, 0.0, 0.0, 0.0, 0.0,
        -1.0002000331878662, -1.0, 0.0, 0.0, -0.020002000033855438, 0.0
    ]

    start = time.time()
    img_arr = p.getCameraImage(pixelWidth,
                               pixelHeight,
                               viewMatrix=viewMatrix,
                               projectionMatrix=projectionMatrix,
                               shadow=1,
                               lightDirection=[1, 1, 1])
    stop = time.time()
    print("renderImage %f" % (stop - start))
    '''