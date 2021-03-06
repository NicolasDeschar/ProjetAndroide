import pybullet as p
import numpy as np
import pybullet_data
import time
from PIL import Image
from pylab import *
from toolbox import discreteProb
import create_file
import random
import toolbox
import classifCNN
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.autograd import Function


class SimpleActionSpace:  # class decrivant l'espace des actions
	def __init__(self, action_list=[], nactions=0):
		if len(action_list) == 0:
			self.actions = np.array([a for a in range(nactions)])
		else:
			self.actions = action_list
		self.size = len(self.actions)

	def sample(self, prob_list=None):
		# renvoie l'action choisie selon la distribution prob_list
		# si prob_list==None, alors l'action est choisie selon une distribution uniforme
		if prob_list is None:
			prob_list = np.ones(self.size)/self.size
		index = discreteProb(prob_list) 
		return self.actions[index]


class CubesEnv:
	def __init__(self,h,nb_cubes,render,urdf_names,widthMin=-3,
			height=3,lengthMin=-3,wMax=3,lMax=3,nb_action=12,gap=2,
			upAxisIndex=2,pixel=320,actions_list=[]):
		self.nn=None #reseau de neurones du classifieur
		self._widthMin=widthMin #largeur minimale de l environnement
		self._height=height     #hauteur de l environnement
		self._lengthMin=lengthMin  #longeur minimale de l env
		self._widthMax=wMax       #largeur maximale
		self._lengthMax=lMax      #longueur maximale
		self._pixel=pixel     #taille de l image capturee par la camera pixel *pixel
		self._cubesPos=[]   #position initiale des cubes
		self.nbCubes=nb_cubes #nombre de cubes dans l env
		self.urdfNames=urdf_names  
		self.nb_states=(wMax-widthMin)*height*(lMax-widthMin)  #nombre d etats
		self.nbAction=nb_action  #nombre d actions
		self.gap=gap #distance minimale entre deux cubes
		self.action_space=SimpleActionSpace(actions_list,nactions=nb_action)

		self.cam_target_pos=[np.random.randint(0,wMax-widthMin)+widthMin,
		np.random.randint(0,lMax-lengthMin)+lengthMin,
		np.random.randint(0,height)]        
		self.cam_dist=1
		self.cam_yaw=np.random.randint(0,360)
		self.cam_roll=np.random.randint(0,360)
		self.cam_pitch=np.random.randint(0,360)
		self.cam_upAxisIndex = upAxisIndex #axe verticale de l env 2 -> axe z
		self._render=render
		self.robotID=[] #id des cubes
		self.h=h #hauteur des cubes
		self.reward=0 #la recompense 
		self.projMatrix = [ 1.0825318098068237, 0.0, 0.0, 0.0, 0.0, 1.732050895690918, 0.0,
		 0.0, 0.0, 0.0, -1.0002000331878662, -1.0, 0.0, 0.0, -0.020002000033855438, 0.0]
		#initialisation des positions
		self._observation=None  
		while(len(self._cubesPos)<nb_cubes):
			pos= [np.random.randint(self.gap,self._widthMax-self._widthMin-self.gap,1)+self._widthMin,
			np.random.randint(self.gap,self._lengthMax-self._lengthMin-self.gap,1)+self._lengthMin,self.h]
			if(self.posDiff(pos)):
				self._cubesPos.append(pos)

		if(self._render):
			p.connect(p.GUI)
		self.reset()

	#deplace la camera selon les paramètres de CubeEnv
	def move_cam(self):
		p.computeViewMatrixFromYawPitchRoll(self.   cam_target_pos,
				self.cam_dist,self.cam_yaw,self.cam_pitch,
				self.cam_roll,self.cam_upAxisIndex)
		p.stepSimulation()

	#place la camera à une position aleatoire selon les limites de l'environnement
	def reset_cam(self,camDistance=1):
		self.cam_target_pos=[
		np.random.randint(0,self._widthMax-self._widthMin)+self._widthMin,
		np.random.randint(0,self._lengthMax-self._lengthMin)+self._lengthMin,
		np.random.randint(0,self._height)]
		self.cam_dist=camDistance
		self.cam_yaw=np.random.randint(0,360)
		self.cam_roll=np.random.randint(0,360)
		self.cam_pitch=np.random.randint(0,360)

	def reset(self):
		create_file.create_plane(self._widthMax-self._widthMin,self._lengthMax-self._lengthMin,self._height) 
		p.resetSimulation()
		p.setGravity(0,0,-10)
		p.loadURDF("plane/plane.urdf", [0, 0, 0])

		robot_id=[]
		for i in range(self.nbCubes):    
			robot_id.append(p.loadURDF(self.urdfNames[i]+".urdf"))
			_ = p.createConstraint(
			robot_id[i],
			-1,#parent link ID (-1 is base)
			-1,#child body ID (-1 is NO body, i.e. a non-dynamic child frame in world coords)
			-1,#child link ID (-1 is base)
			p.JOINT_FIXED,#joint type
			[0, 0, 0],#joint axis in child's frame 
			[0, 0, 0],#parentFramePosition relative to the center of mass frame of the parent
			self._cubesPos[i],#childFramePosition relative to the center of mass frame of the child, or relative to world coordinates if childbodyID is -1
			p.getQuaternionFromEuler([0,0,0]),#parentFrameOrientation relative to parent center of mass coordinates frame
			p.getQuaternionFromEuler([0,0,0]),#childFrameOrientation relatvie to child center of mass or relative to world coords if childBody is -1
			0)#physics client ID
		self.robotID=robot_id
		

	def my_setJointMotorControlMultiDof(self):
		#p.stepSimulation()
		for i in self.robotID:
			p.setJointMotorControlMultiDof(bodyUniqueId=i,
			jointIndex=0,
			controlMode=p.POSITION_CONTROL,
			targetPosition=p.getQuaternionFromEuler((np.random.rand(3)*np.pi).tolist()),
			force=[10,10,10])

	def posDiff(self,pos):
		"""
		renvoie True si les cubes sont suffisamment espacés
		"""
		for i in self._cubesPos:
			if(abs(pos[0]-i[0])<self.gap and abs(pos[1]-i[1])<self.gap):
				return False
		return True

	#renvoie sous la forme np_array l'image de la caméra
	def getExtendedObservation(self,w=320,h=320):
		viewMat = p.computeViewMatrixFromYawPitchRoll(self.cam_target_pos, 
			self.cam_dist, self.cam_yaw, self.cam_pitch, self.cam_roll,self.cam_upAxisIndex)
		img_arr = p.getCameraImage(width=w,height=h,viewMatrix=viewMat,projectionMatrix=self.projMatrix)
		rgb = img_arr[2]
		np_img_arr = np.reshape(rgb, (w, h, 4))
		self._observation = np_img_arr
		return self._observation

	#fonction de calcul de la récompense
	def reward(self):
		obs=self.getExtendedObservation()
		img=toolbox.convert_from_image_to_tensor_gray(obs)
		prob=self.nn(img)
		prob=prob[0].detach().to("cpu").numpy()
		print(prob)
		props=prob[:10]
		if(props[np.argmax(props)]>0.5):
			return props[np.argmax(props)]
		else:
			if(prob[:10].sum()>prob[-1]):
				return prob[:10].sum()/2
			else:
				return 0
	
	#fonction de calcul de la récompense	
	def reward2(self):
		pitchprim=self.cam_pitch
		yawprim=self.cam_yaw
		res=[]
		valeur=0
		for pitch in range(pitchprim-4,pitchprim+4,1):
			self.cam_pitch=pitch
			for yaw in range(yawprim-4,pitchprim+4,1):
				self.cam_yaw=yaw
				obs=self.getExtendedObservation()
				img=toolbox.convert_from_image_to_tensor_gray(obs)
				prob=self.nn(img)
				prob=prob[0].detach().to("cpu").numpy()
				if(prob[np.argmax(prob)]>0.8):
					if(np.argmax(prob)!=10):
						if(np.argmax(prob)>0.9):
							res.append(np.argmax(prob))
							valeur=prob[np.argmax(prob)]
						#print("classe : ",np.argmax(prob),"\nfind the number : ")
						#self._terminated=True
				#return prob[np.argmax(prob)]
		if(len(res)>6 and toolbox.identique(res)):
			print("chiffre trouve ",res[0])
			if(min(res)>0.9):
				self.reward+=1
				return 1
			return np.mean(res)
		else:
			return 0

	#fonction de calcul de la récompense
	def reward3(self):
		obs=self.getExtendedObservation()
		img=toolbox.convert_from_image_to_tensor_gray(obs)
		prob=self.nn(img)
		prob=prob[0].detach().to("cpu").numpy()
		print(prob)
		return max(prob[:10])

	#renvoie True si le cube est correctement detecté
	def termination(self):
		if(reward >0.99):
			return True
		return False

	def stepAlea(self,action): #mouvement alea
		p.stepSimulation()
		viewMat = p.computeViewMatrixFromYawPitchRoll(self.cam_target_pos, self.cam_dist, self.cam_yaw, 
													  self.cam_pitch, self.cam_roll,self.cam_upAxisIndex)
		done = self.termination()
		reward = self.reward()
		return self._observation, reward, done, {}

	#fonction de test de l'environnement
	def testEnv(self):
		for i in range(1000):
			p.stepSimulation()
			self.my_setJointMotorControlMultiDof()
	
	def testCam(self):
		for yaw in range(0, 360, 1):
			self.cam_yaw=yaw
			for pitch in range(0,360,1):
				self.cam_pitch=pitch
				p.stepSimulation()
				time.sleep(1)
				print("reward : ",self.reward())

				

	#réalise un pas de l'environnement en executant l'action action
	def step(self,action,env):
		if action==0:
			env.cam_target_pos[0]+=1
			env.move_cam()
		elif action==1:
			env.cam_target_pos[0]-=1
			env.move_cam()
		elif action==2:
			env.cam_target_pos[1]+=1
			env.move_cam()
		elif action==3:
			env.cam_target_pos[1]-=1
			env.move_cam()
		elif action==4:
			env.cam_target_pos[2]+=1
			env.move_cam()
		elif action==5:
			env.cam_target_pos[2]-=1
			env.move_cam()
		elif action==6:
			env.cam_yaw+=1
			env.move_cam()
		elif action==7:
			env.cam_yaw-=1
			env.move_cam()
		elif action==8:
			env.cam_pitch+=1
			env.move_cam()
		elif action==9:
			env.cam_pitch-=1
			env.move_cam()
		elif action==10:
			env.cam_roll+=1
			env.move_cam()
		elif action==11:
			env.cam_roll-=1
			env.move_cam()
		else :
			pass

		return None,self.reward3(),self.termination(),None
		  
	
	

if(__name__=="__main__"):
	names=create_file.select_cubes({0:1,1:1,2:1,3:1,4:1,5:1,6:1,7:1,8:1,9:1})
	env=CubesEnv(h=2,nb_cubes=7,render=True,urdf_names=names)
	transf=transforms.Compose([transforms.Grayscale(1),transforms.ToTensor(), transforms.Normalize((0.1307,), 	(0.3081,))])
	model=classifCNN.CNN()
	data,loader=classifCNN.create_ClassifDATA(50,transf,"mnist_png/training",True)
	classifCNN.CNNtrain(1,model,loader)
	#print(env.reward(np.ones((28,28))))
	#env.testEnv()
	env.nn=model
	env.testCam()
