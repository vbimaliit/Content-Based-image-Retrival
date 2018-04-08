# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 16:02:38 2017

@author: sunvi

"""
####This is the code to extract the texture feature from the image

import cv2
import numpy as np
from PIL import Image
import mysql.connector
import glob
from skimage import feature
from scipy.spatial import distance
import time
import sys
infile = sys.argv[1]

conn = mysql.connector.connect(host= "localhost",
                  user="root",
                  passwd="root",
                  db="imagefeatures")
x = conn.cursor()

def texture_calculator(img):
    img =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # converting the image into Gray scale to extract the testure feature
    comatrix = feature.greycomatrix(img,[1],[0,np.pi/4,np.pi/2,3*np.pi/4]) # Calculating the co-occurance matrix in the direction 0,45,90,135 degree direction
    ContrastStats = np.reshape(feature.greycoprops(comatrix, 'contrast'),4) # calculating the contrast of the image
    CorrelationtStats = np.reshape(feature.greycoprops(comatrix, 'correlation'),4) # calculating the correlation of the image
    HomogeneityStats = np.reshape(feature.greycoprops(comatrix, 'homogeneity'),4) # calculating the homogeneity of image
    ASMStats = np.reshape(feature.greycoprops(comatrix, 'ASM'),4)# Calculating the ASM of the image

    texture_feature = []


    normalized_contrast = [(i-(np.mean(ContrastStats)))/(np.std(ContrastStats)) for i in ContrastStats] #To normalize - take the value , substract the mean from it and then divide it by its standard deviation  
    
#    normalized_contrast = np.asarray(normalized_contrast)
    
    

    texture_feature.extend(normalized_contrast)

   
    normalised_corr = [(i-(np.mean(CorrelationtStats)))/(np.std(CorrelationtStats)) for i in CorrelationtStats] #To normalize - take the value , substract the mean from it and then divide it by its standard deviation
    texture_feature.extend(normalised_corr)

  
    normalized_homo = [(i-(np.mean(HomogeneityStats)))/(np.std(HomogeneityStats)) for i in HomogeneityStats]#To normalize - take the value , substract the mean from it and then divide it by its standard deviation

#    normalized_homo = np.asarray(normalized_homo)
    texture_feature.extend(normalized_homo)
    
    normalized_asm = [(i-(np.mean(ASMStats)))/(np.std(ASMStats)) for i in ASMStats]#To normalize - take the value , substract the mean from it and then divide it by its standard deviation

#    normalized_asm = np.asarray(normalized_asm)
    texture_feature.extend(normalized_asm)
    return texture_feature
    
def insert_feature(filename,texture_feature): # inserting the texture feature of an image along with its name into Mysql
    try:
        texture_feature = str(texture_feature)
        x.execute("""INSERT INTO texture VALUES (%s,%s)""",(filename,texture_feature))
        
        
    except(mysql.connector.Error, mysql.connector.Warning) as e:
        print(e)
        conn.rollback()
    conn.commit() 
count = 0    
start_time = time.time()
for filename in glob.glob(infile): #assuming jpg
    im=Image.open(filename)
    img = cv2.imread(filename)
    texture = texture_calculator(img)
    insert_feature(filename,texture)
    count = count+1
    im.close()
    
conn.close()
end_time = time.time()
print("Total number of image inserted",count)
print("Total time taken for inserttion" ,start_time-end_time)
