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
        # Draw a random number using probability table p (column vector)
        # Suppose probabilities p=[p(1) ... p(n)] for the values [1:n] are given, sum(p)=1 
        # and the components p(j) are nonnegative. 
        # To generate a random sample of size m from this distribution,
        # imagine that the interval (0,1) is divided into intervals with the lengths p(1),...,p(n).
        # Generate a uniform number rand, if this number falls in the jth interval given the discrete distribution,
        # return the value j. Repeat m times.
        r = np.random.random()
        cumprob = np.hstack((np.zeros(1), np.cumsum(p)))
        sample = -1
        for j in range(p.size):
            if (r > cumprob[j]) & (r <= cumprob[j+1]):
                sample = j
                break
        return sample


def random_policy(env):
    # Returns a random policy given an environnement env
    # Inputs :
    # - env : theenvironnement
    # Output :
    # - pol : the policy

    rand = np.random
    pol = np.zeros(env.nb_states, dtype=np.int16)
    for x in range(env.nb_states):
        pol[x] = rand.choice(env.action_space.actions)
    return pol

def saveIMG(imgs,path="ImageCam/test"):
    """
    Input:
    imgs est une liste de np.array, chaque np.array est une image
    Output:
    rien, image stocker dans path
    """
    try:
        os.makedirs(path)
    except FileExistsError:
        pass

    for i in range(len(imgs)):
        im = Image.fromarray(imgs[i]).convert('L')
        cover=resizeimage.resize_cover(im, [28, 28])
        cover.save(os.path.join(path,str(i)+".png"))


def deleteIMG(path="ImageCam/test"):
    try:
        shutil.rmtree(path)
    except OSError as e:
        print(f"Error:{ e.strerror}")

def create_dataCNN(path="ImageCam/test",trans=transf):
    """
    transformer les images en type du 
    image est le data a donner au CNN
    """
    imgs=ImageFolder(root=path,transform=trans)
    loader=torch.utils.data.DataLoader(imgs,batch_size=1,shuffle=False)
    i,(images,label)=next(enumerate(loader))
    return images,loader

def getOutputCNN(RN,batch_size,images):
    if(batch_size==1):
        return RN(images).detach().numpy()
    else:
        output=RN(images) #x contient batch_size images

def getRewardProb(RN,img,path="ImageCam/test",batch_size=1) :
    saveIMG(img)
    images,loader=create_dataCNN()
    prob=getOutputCNN(RN,batch_size,images)
    deleteIMG()
    return prob


def getListOutputCNN(output):
    res=[]
    for i,prob in enumerate(output): #ici je parcours les images dans le output il y 50 resultats
        res.append(prob.numpy())
    return res
               

def convert_from_image_to_tensor(img):
    """
    input :
    img : une image ou un np.array(image) , torch.from_numpy(image)
    """
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, ), (0.5, ))])
    temp=transform(img)
    return temp.unsqueeze(0)

def convert_from_image_to_tensor_gray(img):
    """
    input :
    img : une image ou un np.array(image) , torch.from_numpy(image)
    """
    transform=transforms.Compose([transforms.Grayscale(1),transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    temp=transform(img)
    return temp.unsqueeze(1)


#img=[np.array(Image.open("cubes/img/0/443.png")),np.array(Image.open("cubes/img/0/8221.png"))]
#saveIMG(img)
#images,loader=create_dataCNN()
#prob=getOutputCNN(RN,batch_size,images) #dans ton cas c'est un tensor comme une liste pour le parcourir regarde la zone en dessous
#deleteIMG()











#print(np.array(Image.open("camera_images/pitch19.png")).shape)

#k=np.array(Image.open("camera_images/pitch19.png"))
#print((k[:,:,0]+k[:,:,1]+k[:,:,2]).shape)
#print(np.array(Image.open("mnist_png/testing/2/1.png")))
#import cv2
#img = cv2.imread('ft/ft0', 0) 
#print(type(img))
#import png
"""
img=np.ones((28,28),dtype="int")
k=[i.tolist() for i in img]
png.from_array(k, 'L').save("smaly.png")


#im=Image.open("camera_images/pitch20.png")

#im.save('{}-{}x{}{}'.format("new_filename", 28, 28, ".png"), im.format)
cover = resizeimage.resize_cover(im, [28, 28])
print(np.array(cover).shape)
#cover.save('test-image-cover.png', im.format)
"""
"""
plt.figure(figsize=(img.shape[1],img.shape[0]))
plt.axis("off")
plt.gcf().subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, wspace = 0, hspace = 0)
plt.savefig("tt.png")
im = Image.open('image.jpg')
im.save('image2.png', 'PNG')

from PIL import Image
im = Image.fromarray(A)
im.save("your_file.jpeg")

"""
#Image.fromarray(img).save("image_entree.jpeg")
#print(Image.Image.convert(mode="L",matrix=img).shape)

#print(np.array(Image.open("mnist_png/testing/2/1.png")))

#mnist2=Image.open("ft/magazine-unlock-01-2.3.1152-_82D7E759A7CE59589C97850534E61A53.jpg").convert('L')
#cover=resizeimage.resize_cover(mnist2, [28, 28])
#cover=np.array(cover).reshape(28*28)
#print(cover.shape,cover[0],len(cover),chr(47))
"""
import cv2
cv2.imshow('image',np.array(Image.open("mnist_png/testing/2/1.png")))
if cv2.waitKey()==2:
    print("j")

"""
    