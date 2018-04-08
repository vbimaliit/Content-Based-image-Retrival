# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 16:32:08 2017

@author: sunvi
"""

import cv2
import numpy as np
from PIL import Image
import mysql.connector
from skimage import feature
from scipy.spatial import distance
import math
import operator
import PIL
import sys
import time
conn = mysql.connector.connect(host= "localhost",
                  user="root",
                  passwd="root",
                  db="imagefeatures")
x = conn.cursor()
def single_histogram(img): #Calculating the histogram of the full image
    img =cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hstm = cv2.calcHist([img], [0, 1,2],None,[16,32,1],[0, 180, 0, 256,0,256])
    hstgm =  cv2.normalize(hstm,hstm)
    return hstgm.flatten()

def texture_calculator(img):
    img =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    comatrix = feature.greycomatrix(img,[1],[0,np.pi/4,np.pi/2,3*np.pi/4])
    ContrastStats = np.reshape(feature.greycoprops(comatrix, 'contrast'),4)
    CorrelationtStats = np.reshape(feature.greycoprops(comatrix, 'correlation'),4)
    HomogeneityStats = np.reshape(feature.greycoprops(comatrix, 'homogeneity'),4)
    ASMStats = np.reshape(feature.greycoprops(comatrix, 'ASM'),4)

    texture_feature = []


    normalized_contrast = [(i-(np.mean(ContrastStats)))/(np.std(ContrastStats)) for i in ContrastStats]
    
#    normalized_contrast = np.asarray(normalized_contrast)
    
    

    texture_feature.extend(normalized_contrast)

   
    normalised_corr = [(i-(np.mean(CorrelationtStats)))/(np.std(CorrelationtStats)) for i in CorrelationtStats]
    texture_feature.extend(normalised_corr)

  
    normalized_homo = [(i-(np.mean(HomogeneityStats)))/(np.std(HomogeneityStats)) for i in HomogeneityStats]

#    normalized_homo = np.asarray(normalized_homo)
    texture_feature.extend(normalized_homo)
    
    normalized_asm = [(i-(np.mean(ASMStats)))/(np.std(ASMStats)) for i in ASMStats]

#    normalized_asm = np.asarray(normalized_asm)
    texture_feature.extend(normalized_asm)
    return texture_feature


def color_extraction(img):
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
    




def read_single_histogram_features_from_Index():
    conn = mysql.connector.connect(host= "localhost",
                  user="root",
                  passwd="root",
                  db="imagefeatures")
    x = conn.cursor()
    sql="select filename,Histogram from fullhistogram;"
    try:
        x.execute(sql)
        data = x.fetchall()
    except(mysql.connector.Error, mysql.connector.Warning) as e:
        print(e)
        conn.rollback()
    conn.close()
    return data

def read_multiple_histogram_features_from_Index():
    conn = mysql.connector.connect(host= "localhost",
                  user="root",
                  passwd="root",
                  db="imagefeatures")
    x = conn.cursor()
    sql="select filename,Histogram from blockhistogram;"
    try:
        x.execute(sql)
        data = x.fetchall()
    except(mysql.connector.Error, mysql.connector.Warning) as e:
        print(e)
        conn.rollback()
    conn.close()
    return data

def read_texture_features_from_Index():
    conn = mysql.connector.connect(host= "localhost",
                  user="root",
                  passwd="root",
                  db="imagefeatures")
    x = conn.cursor()
    sql="select filename,texture_feature from texture;"
    try:
        x.execute(sql)
        data = x.fetchall()
    except(mysql.connector.Error, mysql.connector.Warning) as e:
        print(e)
        conn.rollback()
    conn.close()
    return data



def read_color_texture_features_from_Index():
    conn = mysql.connector.connect(host= "localhost",
                  user="root",
                  passwd="root",
                  db="imagefeatures")
    x = conn.cursor()
    sql="select t.filename,b.Histogram,t.texture_feature from  blockhistogram as b,texture as t;"
    try:
        x.execute(sql)
        data = x.fetchall()
    except(mysql.connector.Error, mysql.connector.Warning) as e:
        print(e)
        conn.rollback()
    conn.close()
    return data


def calc_distance(features,color_histo_query): #calculating the eculidean distance of the fetaure vectors
    color_dist = math.sqrt(sum([(x-y)**2 for x,y in zip(features, color_histo_query)]))
    
    return color_dist
    

def calc_distance_text_color(color_multiple_histo_qurey,color_features,texture_feature_query,text_feat):#calculating the distance of texture and color histogram
    
    color_dist = math.sqrt(sum([(x-y)**2 for x,y in zip(color_multiple_histo_qurey, color_features)]))
    
    text_dist  = math.sqrt(sum([(x-y)**2 for x,y in zip(texture_feature_query, text_feat)]))
    
    return (0.5 *color_dist + 0.5*text_dist )



qurey_image = cv2.imread(sys.argv[1]) ### Accepting the qurey image




color_histo_query = single_histogram(qurey_image)
texture_feature_query = texture_calculator(qurey_image)
#################### calculating the histogram of qurey image and compare it with all the 
#################### images in database and display the best 10 matches
start_time = time.time()
histo_data_feature = read_single_histogram_features_from_Index()

single_histo_dist = {}
for fid, feature1 in histo_data_feature:
#    print(type(feature1))
    features = [float(x) for x in feature1.strip('[]').split(',')]

    dist = calc_distance(features ,color_histo_query)
    
    single_histo_dist[fid] = dist
single_histo_dist = sorted(single_histo_dist.items(), key=operator.itemgetter(1))

sorted_image = single_histo_dist[:10]
print("\n\n image for single histogram",single_histo_dist[:10])
image_address = []

for i in sorted_image:
    img_address, img_distance = i
    image_address.append(img_address)


imgs = [PIL.Image.open(i) for i  in image_address]
min_shape = sorted([(np.sum(i.size),i.size) for i  in imgs])[0][1]
imgs_comb = np.hstack((np.asarray(i.resize(min_shape))for i in imgs))
imgs_comb = PIL.Image.fromarray(imgs_comb)
imgs_comb.save('trial.jpg')


image = PIL.Image.open('trial.jpg')

image.show()    
end_time = time.time()   
print("total time take for retirvel based on single histogram",end_time - start_time)


###############################################################################
###############################################################################
###############################################################################
#################### calculating the block histogram of qurey image and compare it with all the 
#################### images in database and display the best 10 matches

start_time = time.time()
color_multiple_histo_qurey = color_extraction(qurey_image)

histo_multiple_data_feature = read_multiple_histogram_features_from_Index()
multiple_histo_dist = {}
for fid, feature1 in histo_multiple_data_feature:
#    print(type(feature1))
    features = [float(x) for x in feature1.strip('[]').split(',')]

    dist = calc_distance(features ,color_multiple_histo_qurey)
    
    multiple_histo_dist[fid] = dist
multiple_histo_dist = sorted(multiple_histo_dist.items(), key=operator.itemgetter(1))

soretd_image_multiple = multiple_histo_dist[:10]
print("\n\n\image for multiple histogram",multiple_histo_dist[:10])

image_address_multiple = []

for i in soretd_image_multiple:
    img_address_multiple, img_distance = i
    image_address_multiple.append(img_address_multiple)

imgs = [PIL.Image.open(i) for i  in image_address_multiple]
min_shape = sorted([(np.sum(i.size),i.size) for i  in imgs])[0][1]
imgs_comb = np.hstack((np.asarray(i.resize(min_shape))for i in imgs))
imgs_comb = PIL.Image.fromarray(imgs_comb)
imgs_comb.save('trial1.jpg')
image = PIL.Image.open('trial1.jpg')
image.show()

end_time = time.time()
print("total time take for retirvel based on block histogram",end_time - start_time)



###############################################################################
###############################################################################
###############################################################################
###############################################################################
#################### calculating the texture of qurey image and compare it with all the 
#################### images in database and display the best 10 matches
start_time = time.time()
texture_feature = read_texture_features_from_Index()
texture_dist = {}
for fid, feature1 in texture_feature:
#    print(type(feature1))
    features = [float(x) for x in feature1.strip('[]').split(',')]

    dist = calc_distance(features ,texture_feature_query)
    
    texture_dist[fid] = dist
texture_dist = sorted(texture_dist.items(), key=operator.itemgetter(1))
soretd_image_texture = texture_dist[:10]
print("\n\n\image for texture histogram",texture_dist[:10])

image_address_texture = []

for i in soretd_image_texture:
    img_address_texture, img_distance = i
    image_address_texture.append(img_address_texture)

imgs = [PIL.Image.open(i) for i  in image_address_texture]
min_shape = sorted([(np.sum(i.size),i.size) for i  in imgs])[0][1]
imgs_comb = np.hstack((np.asarray(i.resize(min_shape))for i in imgs))
imgs_comb = PIL.Image.fromarray(imgs_comb)
imgs_comb.save('trial2.jpg')
image = PIL.Image.open('trial2.jpg')
image.show()
end_time = time.time()
print("total time take for retirvel based on texture",end_time - start_time)
###############################################################################
###############################################################################
###############################################################################
#################### calculating the histogram  and texture of qurey image and compare it with all the 
#################### images in database and display the best 10 matches
start_time = time.time()
color_texture_feature_retirved = read_color_texture_features_from_Index()
color_texture_feature = {}
for fid,color_hist,texture_feats in color_texture_feature_retirved:
    
    color_features = [float(x) for x in color_hist.strip('[]').split(',')]
    
    text_feat = [float(x) for x in texture_feats.strip('[]').split(',')]
    
    dist = calc_distance_text_color(color_multiple_histo_qurey,color_features,texture_feature_query,text_feat)
    
    color_texture_feature[fid] = dist
    
                         
color_texture_feature = sorted(color_texture_feature.items(), key=operator.itemgetter(1))
soretd_image_texture_col = color_texture_feature[:10]
print("\n\n\image for color feature texture histogram",color_texture_feature[:10])


image_address_color_texture = []

for i in soretd_image_texture_col:
    img_address_texture_col, img_distance = i
    image_address_color_texture.append(img_address_texture_col)

imgs = [PIL.Image.open(i) for i  in image_address_color_texture]
min_shape = sorted([(np.sum(i.size),i.size) for i  in imgs])[0][1]
imgs_comb = np.hstack((np.asarray(i.resize(min_shape))for i in imgs))
imgs_comb = PIL.Image.fromarray(imgs_comb)
imgs_comb.save('trial3.jpg')
image = PIL.Image.open('trial3.jpg')
image.show()
end_time = time.time()
print("total time take for retirvel based on texture and color histogram",end-time - start_time)


#==============================================================================
# for i in single_histo_dist[:10]:
#     img_name = i.strip().split(",")
#     print(i)
#==============================================================================
    
conn.close()