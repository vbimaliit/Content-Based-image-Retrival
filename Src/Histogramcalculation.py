
### This code calculates the histogram of entire image  and then dividing the image 
####into 3*3 blocks after converting the image from BGR to HSV
#### and insert the histogram of all the images in the directory into sql in the form of list.

import cv2
import glob
import numpy as np
from PIL import Image
import mysql.connector # Need to import this in order to perform the SQL operations
import time

conn = mysql.connector.connect(host= "localhost",
                  user="root",           ### Id and password for mysql database
                  passwd="root",
                  db="imagefeatures") ### Name of th database
x = conn.cursor()

import sys

infile = sys.argv[1]

def color_extraction_multiple_blocks(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    rows,cols = img.shape[:2]
    windowsize_r = int(cols/3)
    windowsize_c = int(rows/3)
    hstgm1 = []


# Crop out the window and calculate the histogram
    for r in range(0,img.shape[1] - windowsize_r, windowsize_r):
        for c in range(0,img.shape[0] - windowsize_c, windowsize_c):
            Mask = np.zeros(img.shape[:2], dtype = "uint8")
            cv2.rectangle(Mask, (r, c), (r+windowsize_r, c+windowsize_c), 255, -1)
            hist = calculate_histogram(img,Mask)
        
            hstgm1.extend(hist)    
    return hstgm1

def calculate_histogram(img,mask):
    hstgm = cv2.calcHist([img], [0, 1,2],mask,[16,32,1],[0, 180, 0, 256,0,256])
    hstgm =  cv2.normalize(hstgm,hstgm)
    return hstgm.flatten()

def insert_feature_nultiple_blocks(filename,histo): # inserting the histogram of an image along with its name into Mysql
    try:
        histo = str(histo)
        x.execute("""INSERT INTO blockhistogram VALUES (%s,%s)""",(filename,histo))
        
        
    except(mysql.connector.Error, mysql.connector.Warning) as e:
        print(e)
        conn.rollback()
    conn.commit() 

def color_extraction(img): #Calculating the histogram of the full image
    
    img =cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

 
    hstm = cv2.calcHist([img], [0, 1,2],None,[16,32,1],[0, 180, 0, 256,0,256])
    hstgm =  cv2.normalize(hstm,hstm)
    return hstgm.flatten()
    

def insert_feature(filename,features): # inserting the histogram of an image along with its name into Mysql
    try:
        features = str(features)
        x.execute("""INSERT INTO fullhistogram VALUES (%s,%s)""",(filename,features))
        
        
    except(mysql.connector.Error, mysql.connector.Warning) as e:
        print(e)
        conn.rollback()
    conn.commit()    
    
    
    
# reading all the images from the specified directory
print("inserting data into sql started")
starttime = time.time()
count = 0
for filename in glob.glob(infile): #assuming jpg
    im=Image.open(filename)
    img = cv2.imread(filename)
    histo_feature = []
    histo_feature.extend(color_extraction(img))

    insert_feature(filename,histo_feature)
    
    
    histo = color_extraction_multiple_blocks(img)

    insert_feature_nultiple_blocks(filename,histo)
    
    
    im.close()
    count = count + 1
    
conn.close()
end_time = time.time()
print("total number of feature of image extracted and saved in Mysql",count)
print("total time taken to insert %s images is $s",count,end_time-starttime)

