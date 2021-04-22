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


class SimpleActionSpace:  # class describing the action space of the markov decision process
	def __init__(self, action_list=[], nactions=0):
		if len(action_list) == 0:
			self.actions = np.array([a for a in range(nactions)])
		else:
			self.actions = action_list
		self.size = len(self.actions)

	def sample(self, prob_list=None):
		# returns an action drawn according to the prob_list distribution,
		# if the param is not set, then it is drawn from a uniform distribution
		if prob_list is None:
			prob_list = np.ones(self.size)/self.size
		index = discreteProb(prob_list) 
		return self.actions[index]


class CubesEnv:
	def __init__(self,h,nb_cubes,render,urdf_names,widthMin=-15,height=10,lengthMin=-15,wMax=15,lMax=15,nb_action=12,gap=2,upAxisIndex=2,actions_list=[]):
		self.nn=None #neural network for rewards
	
		self._widthMin=widthMin
		self._height=height
		self._lengthMin=lengthMin
		self._widthMax=wMax
		self._lengthMax=lMax
		
		self._cubesPos=[]
		self.nbCubes=nb_cubes
		self.urdfNames=urdf_names
		self.nb_states=(wMax-widthMin)*height*(lMax-widthMin)
		self.nbAction=nb_action
		self.gap=gap #distance between 2 cubes and the environnement
		self.action_space=SimpleActionSpace(actions_list,nactions=nb_action)

		self.cam_target_pos=[np.random.randint(0,wMax-widthMin)+widthMin,
		np.random.randint(0,lMax-lengthMin)+lengthMin,
		np.random.randint(0,height)]
		self.cam_dist=1
		self.cam_yaw=np.random.randint(0,360)
		self.cam_roll=np.random.randint(0,360)
		self.cam_pitch=np.random.randint(0,360)
		self._render=render
		self.robotID=[] #id of the cubes in the environnement
		self.h=h #height of cubes in the environnement
		self.upAxisIndex = upAxisIndex
		self.projMatrix = [ 1.0825318098068237, 0.0, 0.0, 0.0, 0.0, 1.732050895690918, 0.0,
		 0.0, 0.0, 0.0, -1.0002000331878662, -1.0, 0.0, 0.0, -0.020002000033855438, 0.0]
		#create starting positions
		self._observation=None  #an image of the camera
		while(len(self._cubesPos)<nb_cubes):
			pos= [np.random.randint(self.gap,self._widthMax-self._widthMin-self.gap,1)+self._widthMin,
			np.random.randint(self.gap,self._lengthMax-self._lengthMin-self.gap,1)+self._lengthMin,self.h]
			if(self.posDiff(pos)):
				self._cubesPos.append(pos)

		if(self._render):
			p.connect(p.GUI)
		self.reset()
	
	def move_cam(self): #oublie pas le self
		p.computeViewMatrixFromYawPitchRoll(self.cam_target_pos,self.cam_dist,self.cam_yaw,self.cam_pitch,self.cam_roll,self.upAxisIndex)

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
		for i in self.robotID:
			p.setJointMotorControlMultiDof(bodyUniqueId=i,
			jointIndex=0,
			controlMode=p.POSITION_CONTROL,
			targetPosition=p.getQuaternionFromEuler((np.random.rand(3)*np.pi).tolist()),
			force=[10,10,10])

	def posDiff(self,pos):
		"""
		check the gap between two cubes 
		"""
		for i in self._cubesPos:
			if(abs(pos[0]-i[0])<self.gap and abs(pos[1]-i[1])<self.gap):
				return False
		return True

	def getExtendedObservation(self,w=28,h=28):
		viewMat = p.computeViewMatrixFromYawPitchRoll(self.cam_target_pos, self.cam_dist, self.cam_yaw, self.cam_pitch, self.cam_roll,self.upAxisIndex)
		img_arr = p.getCameraImage(width=w,height=h,viewMatrix=viewMat,projectionMatrix=self.projMatrix)
		rgb = img_arr[2]
		np_img_arr = np.reshape(rgb, (w, h, 4))
		self._observation = np_img_arr
		return self._observation

	def reward(self):
		obs=self.getExtendedObservation()
		img=toolbox.convert_from_image_to_tensor_gray(obs)
		prob=self.nn(img)
		if(prob[np.argmax(prob)]>0.5):
			return prob[np.argmax(prob)]
		else:
			if(prob[:11].sum()>prob[-1]):
				return prob[:11].sum()/2
			else:
				return 0

	def termination(self):
		"""
		indique quand l agent a fini de trouver tous les chiffres
		"""
		if():
			return True
		return False

	def getAction(self):
		return self.getTypeAction(np.random.randint(self.nbAction))
	
	def getTypeAction(self,typeAction):
		isForward=np.random.choices([-1,1])
		if(typeAction==0):
			self.cam_target_pos[0]+=isForward

	def stepAlea(self,action):
		p.stepSimulation()
		viewMat = p.computeViewMatrixFromYawPitchRoll(self.cam_target_pos, self.cam_dist, self.cam_yaw, 
													  self.cam_pitch, self.cam_roll,upAxisIndex)
		done = self.termination()
		reward = self.reward()
		return self._observation, reward, done, {}

	def testEnv(self):
		for i in range(100):
			p.stepSimulation()
			self.my_setJointMotorControlMultiDof()
			time.sleep(1./240.)
	
	def testCam(self):
		return 

	def step(self,action): #IL FAUT DONNER L ENVIRONNEMENT
		if action==0:
			self.cam_target_pos[0]+=1
			self.move_cam()
		elif action==1:
			self.cam_target_pos[0]-=1
			self.move_cam()
		elif action==2:
			self.cam_target_pos[1]+=1
			self.move_cam()
		elif action==3:
			self.cam_target_pos[1]-=1
			self.move_cam()
		elif action==4:
			self.cam_target_pos[2]+=1
			self.move_cam()
		elif action==5:
			env.cam_target_pos[2]-=1
			self.move_cam()
		elif action==6:
			env.cam_yaw+=1
			self.move_cam()
		elif action==7:
			env.cam_yaw-=1
			self.move_cam()
		elif action==8:
			env.cam_pitch+=1
			self.move_cam()
		elif action==9:
			env.cam_pitch-=1
			self.move_cam()
		elif action==10:
			env.cam_roll+=1
			self.move_cam()
		elif action==11:
			env.cam_roll-=1
			self.move_cam()
		else :
			pass
		  
	
	

if(__name__=="__main__"):
	names=create_file.select_cubes({0:3,1:4})
	env=CubesEnv(h=2,nb_cubes=7,render=True,urdf_names=names)
	env.testEnv()