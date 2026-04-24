#converting images for analysis in R
#importing custom module for analysis
import metrics as mt
import os

#desired directories
#note that each class should be separated into different directories.
#however, for the fucntion to work, multiple directories should be specified.
#thus, an empty folder is utilized for this task
#the empty folder is called "none"

#Make directories
os.makedirs('data_binary/Platelets', exist_ok=True)
os.makedirs('data_binary/WBC', exist_ok=True)
os.makedirs('data_binary/RBC', exist_ok=True)
os.makedirs('dataset', exist_ok=True)

#----------------------------------------------------------
#Platelets class

plates = ["data/Platelets", "none"]
binary = 'data_binary/'

#name of .txt file
name = 'dataset/plates.txt'

#converting images
mt.BinaryHistTXT(name, plates, binary)
mt.BinaryShapesTXT(name[-0:-4], [binary+plates[0][5:]])

#----------------------------------------------------------
#WBC class

wbc = ["data/WBC", "none"]
binary = 'data_binary/'

#name of .txt file
name = 'dataset/wbc.txt'

#converting images
mt.BinaryHistTXT(name, wbc, binary)
mt.BinaryShapesTXT(name[-0:-4], [binary+wbc[0][5:]])

#----------------------------------------------------------
#RBC class

rbc = ["data/RBC", "none"]
binary = 'data_binary/'

#name of .txt file
name = 'dataset/rbc.txt'

#converting images
mt.BinaryHistTXT(name, rbc, binary)
mt.BinaryShapesTXT(name[-0:-4], [binary+rbc[0][5:]])