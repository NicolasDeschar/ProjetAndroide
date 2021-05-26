import numpy as np
from random import randrange
import time
from PIL import Image
from pylab import *
import matplotlib.pyplot as plt
import os
import torch
from torchvision.datasets import ImageFolder
from torchvision import  transforms
from resizeimage import resizeimage
import torch

from pathlib import Path
import shutil

transf =transforms.Compose([transforms.Grayscale(1),transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

def discreteProb(p):
    # renvoie un nombre aleatoire selon la table de probabilite p
    r = np.random.random()
    cumprob = np.hstack((np.zeros(1), np.cumsum(p)))
    sample = -1
    for j in range(p.size):
        if (r > cumprob[j]) & (r <= cumprob[j+1]):
            sample = j
            break
    return sample


def random_policy(env):
    # renvoie une politique aléatoire pour l'environnement env
    rand = np.random
    pol = np.zeros(env.nb_states, dtype=np.int16)
    for x in range(env.nb_states):
        pol[x] = rand.choice(env.action_space.actions)
    return pol

def saveIMG(imgs,path="ImageCam/test"):
    """
    Entree:
    imgs est une liste de np.array, chaque np.array est une image
    Sortie:
    None, image stockee dans path
    """
    try:
        os.makedirs(path)
    except FileExistsError:
        pass

    for i in range(len(imgs)):
        im = Image.fromarray(imgs[i]).convert('L')
        cover=resizeimage.resize_cover(im, [320, 320])
        cover.save(os.path.join(path,str(i)+".png"))

#supprime le repertoire path
def deleteIMG(path="ImageCam/test"):
    try:
        shutil.rmtree(path)
    except OSError as e:
        print(f"Error:{ e.strerror}")

def create_dataCNN(path="ImageCam/test",trans=transf):
    """
    renvoie un dataset des images contenues dans path
    """
    imgs=ImageFolder(root=path,transform=trans)
    loader=torch.utils.data.DataLoader(imgs,batch_size=1,shuffle=False)
    i,(images,label)=next(enumerate(loader))
    return images,loader

def getOutputCNN(RN,batch_size,images):
    """
    entree : 
    RN : reseau de neurone
    batch_size : nombre d images par classes
    
    sortie : la sortie du reseau pour les images donnees
    renvoie la sortie du reseau de neurones pour les images : images
    """
    if(batch_size==1): 
        return RN(images).detach().numpy()
    else:
        output=RN(images) #x contient batch_size images


def getRewardProb(RN,img,path="ImageCam/test",batch_size=1) :
    saveIMG(img)
    images,loader=create_dataCNN()
    sortie=getOutputCNN(RN,batch_size,images)
    deleteIMG()
    return sortie



               
#convertit une image de la forme np.array en Tensor 
def convert_from_image_to_tensor(img):
    """
    entree :
    img : une image ou un np.array(image) , torch.from_numpy(image)
    """
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, ), (0.5, ))])
    temp=transform(img)
    return temp.unsqueeze(0)

#convertit une image de la forme np.array en Tensor en la passant en niveau de gris
def convert_from_image_to_tensor_gray(imgArray):
    """
    entree :
    img : un np.array(image)
    """
    img=Image.fromarray(imgArray)
    transform=transforms.Compose([transforms.Resize((28,28)),transforms.Grayscale(1),transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    temp=transform(img)
    return temp.unsqueeze(0)

#renvoie True si tous les éléments de res sont identiques
def identique(res):
    for i in res:
	for j in res:
	    if(i!=j):
	        return False
    return True
    
if(__name__=="__main__"):
    convert_from_image_to_tensor_gray(np.ones((28,28)))
    
