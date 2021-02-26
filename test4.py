import pybullet 
import pybullet_data
import time
import numpy as np
import sys
import data_constant #contained the dictionnay of all the pictures 
#data_constant.obj_name is dict[figure]=list[name of the pictures]


pybullet.connect(pybullet.GUI)
pybullet.setGravity(0,0,-10)
pybullet.loadURDF(pybullet_data.getDataPath()+"/plane.urdf", [0, 0, 0])

# constant h : int
h=2
nb_cube=10
#cube positions
cubes_pos=[]
for i in range(nb_cube):
	cubes_pos.append([np.random.randint(-10,10,1),np.random.randint(-10,10,1),h])
urdf_names=[]
for i in data_constant.obj_name:
    urdf_names.append("cubes"+chr(47)+str(i)+"_"+str(data_constant.obj_name[i][0]))

robot_id=[]
for i in range(nb_cube):
    robot_id.append(pybullet.loadURDF("cube_individual.urdf",globalScaling=0.25))
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


def my_setJointMotorControlMultiDof(robot_id):
    for i in robot_id:
        pybullet.setJointMotorControlMultiDof(bodyUniqueId=i,
            jointIndex=0,
            controlMode=pybullet.POSITION_CONTROL,
            targetPosition=pybullet.getQuaternionFromEuler((np.random.rand(3)*np.pi).tolist()),
            #targetPosition=pybullet.getQuaternionFromEuler([2,3,4]),
            force=[10,10,10])

for i in range(10000):
    pybullet.stepSimulation()
    my_setJointMotorControlMultiDof(robot_id)
    time.sleep(1./240.)
    


