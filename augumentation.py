# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 16:44:14 2022

@author: vijay
"""



import tensorflow as tf

#import matplotlib.pyplot as plt
import cv2 
import os
#import random
import numpy as np
import glob
import ntpath
#import time


img_ext ='.png'
img_in_ext = -4

# Processing folder drive 
#====================================================================================
path2drive = r'D:\Machine Learning\DeepUnet_trail'

data_folder = 'Fused Dataset' # all the data 
image_input = "images2"
label_input ="labels2"

image_output = "new_images"
label_output ="new_labels"

path_images = os.path.join(path2drive, data_folder)

path_input_img = os.path.join(path_images,image_input)
path_input_label = os.path.join(path_images,label_input)

'saving the data in folder same name'

path_output_img = os.path.join(path_images,image_output)
path_output_label = os.path.join(path_images,label_output)

if  os.path.exists(path_output_img) == False:
    os.mkdir(path_output_img)
if os.path.exists(path_output_label) == False:
    os.mkdir(path_output_label)


for impath in glob.glob(os.path.join(path_input_img,'*.png')):
    print(ntpath.basename(impath))
    
    img = cv2.imread(str(impath))
     
    
    flip_up_down_name = ntpath.basename(impath)[:img_in_ext]+"_flip_up_down" +img_ext
    flip_up_down_img = tf.image.flip_up_down(img)
    flip_up_down_img =np.array(flip_up_down_img, dtype='uint8')
   
    
   
    flip_left_right_name = ntpath.basename(impath)[:img_in_ext]+"_flip_left_right" +img_ext
    flip_left_right_img = tf.image.flip_left_right(img)
    flip_left_right_img= np.array(flip_left_right_img, dtype='uint8')
    
    
    rot90_name = ntpath.basename(impath)[:img_in_ext]+"_rot90" +img_ext 
    rot_90_img = tf.image.rot90(img, k=1)
    rot_90_img= np.array(rot_90_img, dtype='uint8')
    
    
    rot_180_name = ntpath.basename(impath)[:img_in_ext]+"_rot_180" +img_ext    
    rot_180_img = tf.image.rot90(img, k=2)  
    rot_180_img= np.array(rot_180_img, dtype='uint8')
    
    
    rot_270_name = ntpath.basename(impath)[:img_in_ext]+"_rot_270" +img_ext   
    rot_270_img = tf.image.rot90(img, k=3)  
    rot_270_img= np.array(rot_270_img, dtype='uint8')
    
    bright_name = ntpath.basename(impath)[:img_in_ext]+"_bright" +img_ext
    bright_img = tf.image.adjust_brightness(img, delta=0.15)
    bright_img = np.array(bright_img, dtype='uint8')

    
    contrast_name = ntpath.basename(impath)[:img_in_ext]+"_contrast" +img_ext 
    contrast_img = tf.image.adjust_contrast(img, 2.)
    contrast_img= np.array(contrast_img, dtype='uint8')
   
    
 #=================================================================================  
    
    label_path =os.path.join(path_input_label,ntpath.basename(impath))
    
    label = cv2.imread(str(label_path))
    flip_up_down_label = tf.image.flip_up_down(label)
    flip_up_down_label =np.array(flip_up_down_label, dtype='uint8')
    
    
    flip_left_right_label = tf.image.flip_left_right(label)
    flip_left_right_label= np.array(flip_left_right_label, dtype='uint8')   
    
      
    rot_90_label = tf.image.rot90(label, k=1)
    rot_90_label= np.array(rot_90_label, dtype='uint8')


    rot_180_label = tf.image.rot90(label, k=2)  
    rot_180_label= np.array(rot_180_label, dtype='uint8')

    rot_270_label = tf.image.rot90(label, k=3)  
    rot_270_label= np.array(rot_270_label, dtype='uint8')
    
    
    bright_label = tf.image.adjust_brightness(label, delta=0.15)
    bright_label = np.array(bright_label, dtype='uint8')
    
    
    contrast_label = tf.image.adjust_contrast(label, 2.)
    contrast_label= np.array(contrast_label, dtype='uint8')

 #===================================================================================
    cv2.imwrite(os.path.join(path_output_img, ntpath.basename(impath)), img)
    cv2.imwrite(os.path.join(path_output_label, ntpath.basename(impath)), label)
    
    cv2.imwrite(os.path.join(path_output_img, flip_up_down_name), flip_up_down_img)
    cv2.imwrite(os.path.join(path_output_label, flip_up_down_name), flip_up_down_label)
    
    cv2.imwrite(os.path.join(path_output_img, flip_left_right_name), flip_left_right_img)
    cv2.imwrite(os.path.join(path_output_label, flip_left_right_name), flip_left_right_label)

    cv2.imwrite(os.path.join(path_output_img, rot90_name), rot_90_img)
    cv2.imwrite(os.path.join(path_output_label, rot90_name), rot_90_label)
    
    cv2.imwrite(os.path.join(path_output_img, rot_180_name), rot_180_img)
    cv2.imwrite(os.path.join(path_output_label, rot_180_name), rot_180_label)
    
    cv2.imwrite(os.path.join(path_output_img, rot_270_name), rot_270_img)
    cv2.imwrite(os.path.join(path_output_label, rot_270_name), rot_270_label)
    
    cv2.imwrite(os.path.join(path_output_img, bright_name), bright_img)
    cv2.imwrite(os.path.join(path_output_label, bright_name), bright_label)
    
    cv2.imwrite(os.path.join(path_output_img, contrast_name), contrast_img)
    cv2.imwrite(os.path.join(path_output_label, contrast_name), contrast_label)
    
    











