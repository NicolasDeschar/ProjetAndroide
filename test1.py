import pybullet as p
import time
import pybullet_data
import numpy as np
import init_urdf 

# constant h : int
h=2
nb_cube=10
#cube positions
cubes_pos=[]
for i in range(10):
	cubes_pos.append([np.random.randint(-10,10,1),np.random.randint(-10,10,1),h])
urdf_names=[]

#create urdf files and list of urdf names
for i in range(10):
	init_urdf.create(str(i)+"_0","cubes"+chr(92)+str(i)+"_0")
	urdf_names.append("cubes"+chr(92)+str(i)+"_0"+".urdf")



physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,0)
planeId = p.loadURDF("plane.urdf")

for i in range(nb_cube):
	cubeStartPos = cubes_pos[i]
	x,y,z=np.random.randint(-1,2,1),np.random.randint(-1,2,1),np.random.randint(-1,2,1)
	cubeStartOrientation = p.getQuaternionFromEuler([x,y,z])
	cubId = p.loadURDF(urdf_names[i],cubeStartPos, cubeStartOrientation,globalScaling=0.25)
	cubePos, cubeOrn = p.getBasePositionAndOrientation(cubId)
	print(cubePos,cubeOrn)

for i in range (10000):
	p.stepSimulation()
	time.sleep(1./240.)
p.disconnect()
