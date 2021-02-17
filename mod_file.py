import shutil

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

create_mtl("cubes/9963.mtl","9963.png")
create_obj("cubes/9963.obj","9963.mtl")