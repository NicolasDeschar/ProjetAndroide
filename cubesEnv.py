import pybullet as p
import numpy as np
import pybullet_data
import time
from PIL import Image
from pylab import *
from toolbox import discreteProb
import create_file
import random
import q_learning as q
import toolbox

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
    def __init__(self,NN,h,nb_cubes,render,urdf_names,width=30,height=10,length=30,nb_action=12,gap=2,upAxisIndex=2,actions_list=[]):
        self.nn=NN
        
        self._width=width
        self._heigth=height
        self._length=length
        
        self._cubesPos=[]
        self.nbCubes=nb_cubes
        self.urdfNames=urdf_names
        self.nb_states=width*height*length
        self.nbAction=nb_action
        self.gap=gap #distance between 2 cubes and the environnement
        self.action_space=SimpleActionSpace(actions_list,nactions=nb_action)

        self.cam_target_pos=[np.random.randint(gap,width-gap),np.random.randint(gap,length-gap),
        np.random.randint(gap,height-gap)]
        self.cam_dist=1
        self.cam_yaw=np.random.randint(0,360)
        self.cam_roll=np.random.randint(0,360)
        self.cam_pitch=np.random.randint(0,360)
        self._render=render
        self._p=p
        self.robotID=[] #id of the cubes in the environnement
        self.h=h #height of cubes in the environnement
        #creer la position initiale
        self._observation=None  #a image of the camera
        self.upAxisIndex = upAxisIndex
        self.projMatrix = [ 1.0825318098068237, 0.0, 0.0, 0.0, 0.0, 1.732050895690918, 0.0,
         0.0, 0.0, 0.0, -1.0002000331878662, -1.0, 0.0, 0.0, -0.020002000033855438, 0.0]
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
        """
        check the gap between two cubes 
        """
        for i in self._cubesPos:
            if(abs(pos[0]-i[0])<self.gap and abs(pos[1]-i[1])<self.gap):
                return False
        return True

    def getExtendedObservation(self):
        viewMat = p.computeViewMatrixFromYawPitchRoll(self.cam_target_pos, self.cam_dist, self.cam_yaw, self.cam_pitch, self.cam_roll,
                             upAxisIndex)
        img_arr = p.getCameraImage(width=self._width,
                               height=self._height,
                               viewMatrix=viewMat,
                               projectionMatrix=self.projMatrix)
        rgb = img_arr[2]
        np_img_arr = np.reshape(rgb, (self._height, self._width, 4))
        self._observation = np_img_arr
        return self._observation

#a definir
    def reward(self):
        obs=self.getExtendedObservation()
        return toolbox.getRewardProb(self.nn,obs)
         

    def termination(self):
        """
        indique quand l agent a fini de trouver tous les chiffres
        """
        if():
            return True
        return False

    def getAction(self):
        return getTypeAction(np.random.randint(self.nbAction))
    
    def getTypeAction(self,typeAction):
        isForward=np.random.choices([-1,1])
        if(typeAction==0):
            self.cam_target_pos[0]+=isForward

    def stepAlea(self,action):
        p.stepSimulation()
        viewMat = p.computeViewMatrixFromYawPitchRoll(self.cam_target_pos, self.cam_dist, 
        self.cam_yaw, self.cam_pitch, self.cam_roll,
                             upAxisIndex)
        done = self.termination()
        reward = self.reward()
        return self._observation, reward, done, {}


def start_env():
    names=create_file.select_cubes({0:3,1:4})
    CubesEnv(h=2,nb_cubes=7,render=True,urdf_names=names)

def reset_env():
    the_cam=cam([0,0,0],0,0,0)
    CubesEnv.reset()


def stop():
    CubesEnv.close()


class cam:
    def __init__(self,cam_pos, yaw,pitch,roll):
        self.cam_pos=cam_pos
        self.yaw=yaw
        self.pitch=pitch
        self.roll=roll
        self.cam_distance=2
        self.upAxisIndex=2

    def move_cam():
        CubesEnv.computeViewMatrixFromYawPitchRoll(self.cam_pos,self.cam_distance,self.yaw,self.pitch,self.roll,self.upAxisIndex)



def step(action):

    if action==0:
        the_cam.cam_pos[0]+=1
        the_cam.move_cam()
    elif action==1:
        the_cam.cam_pos[0]-=1
        the_cam.move_cam()
    elif action==2:
        the_cam.cam_pos[1]+=1
        the_cam.move_cam()
    elif action==3:
        the_cam.cam_pos[1]-=1
        the_cam.move_cam()
    elif action==4:
        the_cam.cam_pos[2]+=1
        the_cam.move_cam()
    elif action==5:
        the_cam.cam_pos[2]-=1
        the_cam.move_cam()
    elif action==6:
        the_cam.yaw+=1
        the_cam.move_cam()
    elif action==7:
        the_cam.yaw-=1
        the_cam.move_cam()
    elif action==8:
        the_cam.pitch+=1
        the_cam.move_cam()
    elif action==9:
        the_cam.pitch-=1
        the_cam.move_cam()
    elif action==10:
        the_cam.roll+=1
        the_cam.move_cam()
    elif action==11:
        the_cam.roll-=1
        the_cam.move_cam()
    else :
        pass


def get_image():
    # TODO
    pass 



'''
def main():
    names=create_file.select_cubes({0:3,1:4})
    CubesEnv(h=2,nb_cubes=7,render=True,urdf_names=names)
    done = False
    while (not done):
        action = []
        state, reward, done, info = CubesEnv.step(action)
        obs = CubesEnv.getExtendedObservation()
'''

def learning_simu(episodes=100,itera=5000,eps=0.01,a=0.1, lr=0.01):
    the_cam=cam([0,0,0],0,0,0)
    q.deep_Q(episodes,itera,eps,a,lr)
