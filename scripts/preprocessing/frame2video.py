import cv2
import numpy as np
import glob
import os
import time
import logging

'''
CONVERT ALL FRAMES IN A FOLDER INTO A VIDEO.
'''

abs_path_to_dataset = "..."
folderIn = abs_path_to_dataset + "/20bn-jester-v1/"
folderOut = abs_path_to_dataset + "/videos/"

# Create videos folder @{folderOut} if not exist
if not os.path.exists(folderOut):
    os.makedirs(folderOut)

directories = [f for f in os.scandir(folderIn) if f.is_dir()]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

logger = logging.getLogger('logger')
logger.setLevel(logging.INFO)
vid_s_time = time.time()

for num, directory in enumerate(directories):

    img_array = []
    if os.path.exists(folderOut + directory.name+'.mp4'):
        continue
    for filename in sorted(glob.glob(directory.path+'/*.jpg')):
        img = cv2.imread(filename)
        img_array.append(img)

    height, width, layers = img.shape
    size = (width,height)
    out = cv2.VideoWriter(folderOut + directory.name+'.mp4', fourcc, 12.0, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    if num > 0 and num % log_int == 0:
        vid_e_time = time.time()
        loop_t = vid_e_time - vid_s_time
        vid_s_time = time.time()
        t = (len(directories) - num)/log_int * loop_t /3600
        logger.info('%04d/%04d is done (%4.2f%%). Time for processing %02d videos: %4.2f seconds. ETA: %4.2fh' % (num, len(directories),num/len(directories)*100,log_int, loop_t, t))
