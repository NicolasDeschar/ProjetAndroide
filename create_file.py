import shutil
import os
import numpy as np


def createNames(pathImg="cubes/img"):
	"""
	pathImg doit contenir que des images
	renvoie la liste des noms des elements de pathImg
	"""
	obj_name={}
	for i in range(10):
		path =pathImg+chr(47)+str(i)
		png_name=[]
		files = os.listdir(path)
		for name in files:
			png_name.append(name.replace(".png",""))
		obj_name[i]=png_name
	return obj_name

def create_mtl(destination,photo):
	"""
	entree : 
	photo : texture a mettre sur une des faces du cube
	destination est le chemin ou enregistrer ce fichier
	
	creer le fichier de texture pour les cubes
	"""
	shutil.copyfile("file_mtl.mtl",destination)
	f=open(destination,"a")
	f.write("\n  map_Kd "+photo)
	f.close()

def create_obj(destination,mtl_name):
	"""
	creer la forme de la cube en integrant la texture
	"""
	shutil.copyfile("file_cube.obj",destination)
	f=open(destination,"r")
	lines=f.readlines()
	lines[0]="mtllib "+mtl_name+"\n"
	f.close()
	f=open(destination,"w")
	f.writelines(lines)
	f.close()

def create_urdf(destination, objet):
	"""
	creer le fichier urdf pour la cube object
	"""
	shutil.copyfile("cube_individual.urdf",destination)
	f=open(destination,"r")
	lines=f.readlines()
	lines[19]="         <mesh filename='"+objet+"' scale='1 1 1'/>"
	f.close()
	f=open(destination,"w")
	f.writelines(lines)
	f.close()	



def create(path="cubes",pathIm="cubes/img"):
	"""
	creer tous les fichiers urdf mtl et obj pour les images du dossier 
	"""
	obj_name=createNames(pathImg=pathIm)
	for i in obj_name:
		for j in obj_name[i]:
			dest=path+chr(47)+str(i)+"_"+str(j)
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


def select_cubes(dic):
	"""select nb names of cube, figure is given by dic"""
	name=[]
	obj_name=create()
	for number in dic:
		for j in range(dic[number]):
			nbElt=len(obj_name[number])
			name.append("cubes"+chr(47)+str(number)+"_"+obj_name[number][np.random.randint(0,nbElt)])
	return name

