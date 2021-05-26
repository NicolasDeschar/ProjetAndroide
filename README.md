# ProjetAndroide
Développement d'environnements Pybullet Gym pour l'évaluation de politiques de perception active

## Pour lancer le projet :

  Installer les packages
  
  -pytorch
  
  -torchvision
  
  -python-resize-image
  
  -Pillow
  
  -pylab
  
  -pybullet
  
  -gym 
  
  
  
  Extraire les bases d'apprentissage du classifieur contenues dans mnist_png.zip.001 et mnist_png.zip.002


Executer projet.py


## Description des fichiers :

Q_learning.py contient les fonctions nécessaires à l'algorithme deep-Q

classifCNN.py contient les fonctions nécessaires au classifieur à réseau de neurones convolutifs

classifMV.py contient les fonctions nécessaire au classifieur à maximum de vraisemblance

classifRNPR.py contient les fonctions nécessaires au classifieur perceptron multicouche à rétropropagation

create_file.py contient les fonctions nécessaires pour créer les fichiers .urdf, .mtl, .obj et ainsi obtenir les cubes de l'environnement.

creer_Env.py contient les fonctions initialisant un environnement

cubesEnv.py contient les fonctions créant les cubes et gèrant la caméra

projet.py lance l'ensemble du projet, donc l'algorithme deep-Q

test.py est un executable générant un environnement contenant des cubes

## Description des Dossiers :
Cubes contient un ensemble de fichiers .urdf, .mtl, .obj utilisable pour les placer dans l'environnement pybullet et un sous ensemble d'images de la base MNIST dans le dossier img

plane contient les fichiers pour personnaliser l'environnement (choisir la taille de l'environnement)

