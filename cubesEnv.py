import pybullet as p
import numpy as np
import pybullet_data
import time
from PIL import Image
from pylab import *
from toolbox import discreteProb
import create_file

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
    def __init__(self,h,nb_cubes,render,urdf_names,width=30,height=10,length=30,nb_action=12,actions_list=[]):
        self._width=width
        self._heigth=height
        self._length=length
        self._cubesPos=[]
        self.nbCubes=nb_cubes
        self.urdfNames=urdf_names
        self.nb_states=width*height*length
        self.action_space=SimpleActionSpace(actions_list,nactions=nb_action)
        self.cam_target_pos=[np.random.randint(2,width),np.random.randint(2,length),
        np.random.randint(2,height)]
        self.cam_dist=1
        self.cam_yaw=np.random.randint(0,360)
        self.cam_roll=np.random.randint(0,360)
        self.cam_pitch=np.random.randint(0,360)
        self._render=render
        self._p=p
        self.robotID=[]
        self.h=h #height of cubes in the environnement
        #creer la position initiale
        while(len(self._cubesPos)<nb_cubes):
            pos= [np.random.randint(0,self._width,1),np.random.randint(0,self._length,1),self.h]
            if(self.posDiff(pos)):
	            self._cubesPos.append(pos)
    
        if(self._render):
            p.connect(p.GUI)
        self.reset()
    



    def reset(self):
        create_file.create_plane(self._width,self._length,self._heigth) 
        p.resetSimulation()
        p.setGravity(0,0,-10)
        p.loadURDF("plane/plane.urdf", [0, 0, 0])

        robot_id=[]
        for i in range(self.nbCubes):    
            robot_id.append(p.loadURDF(self.urdfNames[i]+".urdf"))
            _ = self._p.createConstraint(
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
        for i in range(10000):
            p.stepSimulation()
            self.my_setJointMotorControlMultiDof()
            time.sleep(1./240.)


    def my_setJointMotorControlMultiDof(self):
        for i in self.robotID:
            p.setJointMotorControlMultiDof(bodyUniqueId=i,
            jointIndex=0,
            controlMode=p.POSITION_CONTROL,
            targetPosition=p.getQuaternionFromEuler((np.random.rand(3)*np.pi).tolist()),
            force=[10,10,10])

    def posDiff(self,pos):
        for i in self._cubesPos:
            if(abs(pos[0]-i[0])<2 and abs(pos[1]-i[1])<2):
                return False
        return True



def main():
    names=create_file.select_cubes({0:3,1:4})
    CubesEnv(h=2,nb_cubes=7,render=True,urdf_names=names)

main()
