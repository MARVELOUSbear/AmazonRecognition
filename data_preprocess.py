import os

import numpy as np
import tensorflow as tf

from PIL import Image

dir = "./train-jpg/"
width = 64
height = 64
l = width*height

train_jpg_list = os.listdir(dir)


input_count = len(train_jpg_list)
cut_num = 250
remain = input_count % cut_num
split = input_count // cut_num
del train_jpg_list

def ImageToArray(file,width,height):
    '''3 color channels, 1*length'''
    img_raw = Image.open(file)
    img = img_raw.resize((width, height), Image.ANTIALIAS)
    width,height = img.size
    data = img.convert("RGB").getdata()
    data = np.array(data, dtype='int')
    result = np.reshape(data,(1,width*height*3))
    return result

for j in range(split):
    if j+1 == split:
        count = cut_num + remain
    else: count = cut_num
    input_images = np.array([[0]*l*3 for i in range(count)])
    for i in range(j*cut_num,j*cut_num+count):
        file = dir + "train_%s.jpg" % i
        data = ImageToArray(file,width,height)
        if i % 10 == 0: print("Have completed " + str(i))
        input_images[i-j*cut_num] = data
    np.savetxt( "train-jpg-all-"+str(j)+".txt",input_images,fmt="%d")



