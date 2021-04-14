import pybullet as p
import numpy as np
import pybullet_data
import time
from PIL import Image
from pylab import *
from toolbox import discreteProb
import create_file
import random
import Q_learning as q
import toolbox
import classifCNN
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.autograd import Function
import cubesEnv as CE

def start_env():
    transf=transforms.Compose([transforms.Grayscale(1),transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    model=classifCNN.CNN()
    data,loader=classifCNN.create_ClassifDATA(50,transf,"mnist_png/training",True)
    ct,tt,st=classifCNN.CNNtrain(2,model,loader) #entrainement du model

    names=create_file.select_cubes({0:3,1:4})
    env=CE.CubesEnv(h=2,nb_cubes=7,render=True,urdf_names=names)
    env.nn=model
    return env

def test_cam():
    p.computeViewMatrixFromYawPitchRoll([10,10,10],2,5,-15,10,2)

"""
def reset_env():
    CubesEnv.cam_target_pos=[np.random.randint(gap,width-gap),np.random.randint(gap,length-gap),np.random.randint(gap,height-gap)]
    self.cam_dist=1
    self.cam_yaw=np.random.randint(0,360)
    self.cam_roll=np.random.randint(0,360)
    self.cam_pitch=np.random.randint(0,360)
    
    CubesEnv.reset()
"""
def reset_env(env):
    env.reset_cam() 
    env.reset()


def stop(env):
    env.close() #CHANGER et pas CubesEnv


def get_image(env):
    return env.getExtendedObservation()

def learning_simu(episodes=100,itera=5000,eps=0.01,a=0.1, lr=0.01):
    names=create_file.select_cubes({0:3,1:4})
    env=CE.CubesEnv(h=2,nb_cubes=7,render=True,urdf_names=names)
    q.deep_Q(episodes,itera,eps,a,lr)