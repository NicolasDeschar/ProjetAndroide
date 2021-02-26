import shutil
import data_constant

def create_mtl(destination,photo):
	shutil.copyfile("file_mtl.mtl",destination)
	f=open(destination,"a")
	f.write("map_Kd "+photo)
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
	
for i in data_constant.obj_name:
	for j in data_constant.obj_name[i]:
		dest="cubes"+chr(47)+str(i)+"_"+str(j)
		create_mtl(dest+".mtl","cubes/img"+chr(47)+str(i)+chr(47)+str(j)+".png")
		create_obj(dest+".obj",str(i)+"_"+str(j)+".mtl")
		create_urdf(dest+".urdf",dest+".obj")
	