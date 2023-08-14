import numpy as np
import os
import glob
import shutil
from score_utils import mean_score


# path to the images and the text file which holds the scores and ids
base_images_path = '/home/bkcs/NIMA/archive/images/images/'
ava_dataset_path = '/home/bkcs/NIMA/AVA.txt'
dst_dir = '/home/bkcs/NIMA/test_img_4-5, 6-7/'




print("Loading training set and val set")
with open(ava_dataset_path, mode='r') as f:
    lines = f.readlines()
    #print(lines)
    for i, line in enumerate(lines):
        token = line.split()
        id = int(token[1])
        #print(id)

        values = np.array(token[2:12], dtype='float32')
        values /= values.sum()
        mean = mean_score(values)

        file_path = base_images_path + str(id) + '.jpg'
        if (mean > 0 and mean < 4): ## 1<= mean <=4
            if os.path.exists(file_path):
               #print(file_path)
               shutil.copy(file_path, dst_dir + "Low")
        elif (mean >= 4 and mean <5):
            if os.path.exists(file_path):
               #print(file_path)
               shutil.copy(file_path, dst_dir)
        elif (mean >= 6 and mean <7):
            if os.path.exists(file_path):
               #print(file_path)
               shutil.copy(file_path, dst_dir)         
        elif (mean > 7 and mean <= 10):
             if os.path.exists(file_path):
               #print(file_path)
               shutil.copy(file_path, dst_dir + "High")
        count = 255000 // 20
        if i % count == 0 and i != 0:
            print('Loaded %d percent of the dataset' % (i / 255000. * 100))
