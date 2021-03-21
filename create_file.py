import shutil
import os


#contains the names of all the images
def createNames():
	obj_name={}
	for i in range(10):
		path ="cubes/img"+chr(47)+str(i)
		png_name=[]
		files = os.listdir(path)
		for name in files:
			png_name.append(name.replace(".png",""))
		obj_name[i]=png_name
	return obj_name

def create_mtl(destination,photo):
	shutil.copyfile("file_mtl.mtl",destination)
	f=open(destination,"a")
	f.write("\n  map_Kd "+photo)
	f.close()

def create_obj(destination,mtl_name):
	shutil.copyfile("file_cube.obj",destination)
	f=open(destination,"r")
	lines=f.readlines()
	lines[0]="mtllib "+mtl_name+"\n"
	f.close()
	f=open(destination,"w")
	f.writelines(lines)
	f.close()

def create_urdf(destination,object):
	shutil.copyfile("cube_individual.urdf",destination)
	f=open(destination,"r")
	lines=f.readlines()
	lines[19]="         <mesh filename='"+object+"' scale='1 1 1'/>"
	f.close()
	f=open(destination,"w")
	f.writelines(lines)
	f.close()	



def create():
	obj_name=createNames()
	for i in obj_name:
		for j in obj_name[i]:
			dest="cubes"+chr(47)+str(i)+"_"+str(j)
			create_mtl(dest+".mtl","img"+chr(47)+str(i)+chr(47)+str(j)+".png")
			create_obj(dest+".obj",str(i)+"_"+str(j)+".mtl")
			create_urdf(dest+".urdf",dest+".obj")
	return obj_name


def create_plane(W,H,L,destination="plane/plane.urdf"):
	f=open(destination,"r")
	lines=f.readlines()
	lines[23]="	 	    <box size='"+str(W)+" "+str(L)+" "+str(H)+"'/>\n"
	f.close()
	f=open(destination,"w")
	f.writelines(lines)
	f.close()	
