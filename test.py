#exemple de tests pour obtenir un environnement contenant des cubes

import pybullet 
import pybullet_data
import time
import numpy as np
import sys
import create_file

#taille de l environnement
W=100 
H=100
L=10


def posDiff(pos,list_pos): #pour obtenir des positions de cubes espaces
    for i in list_pos:
        if(abs(pos[0]-i[0])<1.5 and abs(pos[1]-i[1])<1.5):
            return False
    return True

def my_setJointMotorControlMultiDof(robot_id): #creer 
    for i in robot_id:
        pybullet.setJointMotorControlMultiDof(bodyUniqueId=i,
            jointIndex=0,
            controlMode=pybullet.POSITION_CONTROL,
            targetPosition=pybullet.getQuaternionFromEuler((np.random.rand(3)*np.pi).tolist()),
            force=[10,10,10])

#creation de l environnement
create_file.create_plane(W,L,H)        
pybullet.connect(pybullet.GUI)
pybullet.setGravity(0,0,-10)
#pybullet.loadURDF(pybullet_data.getDataPath()+"/plane.urdf", [0, 0, 0])
pybullet.loadURDF("plane/plane.urdf",[0,0,0])
# constante h : int
h=2
nb_cube=10

#cube positions
cubes_pos=[]
while(len(cubes_pos)<nb_cube):
    pos= [np.random.randint(-10,10,1),np.random.randint(-10,10,1),h]
    if(posDiff(pos,cubes_pos)):
	    cubes_pos.append(pos)
	
#creation des cubes
urdf_names=[]
nameObj=create_file.create() #create all the 
for i in nameObj:
    urdf_names.append("cubes"+chr(47)+str(i)+"_"+str(nameObj[i][0]))

robot_id=[]
#creer un referentiel pour chaque cube, pour qu il reste en hauteur sans tomber de l environnement
for i in range(nb_cube):
    robot_id.append(pybullet.loadURDF(urdf_names[i]+".urdf"))
    #robot_id.append(pybullet.loadURDF("cube_individual.urdf"))
    _ = pybullet.createConstraint(
        robot_id[i],
        -1,#parent link ID (-1 is base)
        -1,#child body ID (-1 is NO body, i.e. a non-dynamic child frame in world coords)
        -1,#child link ID (-1 is base)
        pybullet.JOINT_FIXED,#joint type
        [0, 0, 0],#joint axis in child's frame 
        [0, 0, 0],#parentFramePosition relative to the center of mass frame of the parent
        cubes_pos[i],#childFramePosition relative to the center of mass frame of the child, or relative to world coordinates if childbodyID is -1
        pybullet.getQuaternionFromEuler([0,0,0]),#parentFrameOrientation relative to parent center of mass coordinates frame
        pybullet.getQuaternionFromEuler([0,0,0]),#childFrameOrientation relatvie to child center of mass or relative to world coords if childBody is -1
        0)#physics client ID



for i in range(10000):
    pybullet.stepSimulation()
    my_setJointMotorControlMultiDof(robot_id)
    time.sleep(1./240.)
    


