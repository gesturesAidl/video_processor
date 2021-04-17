import cv2
import numpy as np
import glob
import os
import time
import logging
import random
from imgaug import augmenters as iaa

folderIn  = './JESTER_DATASET/20bn-jester-v1'             # Path for dataset
folderOut = './JESTER_DATASET/20bn-jester-v1/augmented'   # Path for output augmented images   

if not os.path.exists(folderOut):
    os.makedirs(folderOut)

directories = [f for f in os.scandir(folderIn) if f.is_dir() ]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

logging.basicConfig(filename='frameaug2video.log')
log_int = 50
logger = logging.getLogger('logger')
logger.setLevel(logging.INFO)
vid_s_time = time.time()

# Read validation videos numbers to skip them, since data augmentation is only done in training data
validation_path = r"C:\JESTER_DATASET\jester_dataset_csv_9_classes\validation.csv"
val_videos = np.genfromtxt(validation_path, dtype='int', delimiter=",", skip_header=1, usecols=0)  # reads csv
val_videos_list = list(val_videos)

for num, directory in enumerate(directories):

    if (int(directory.name)) not in val_videos_list:     # only do data augmentation in training data

        img_array = []
        
        '''
        # Trial I
        contrast = random.uniform(0.4, 1.6)        # random contrast value per video
        brightness = random.uniform(0.5, 1.5)      # random brightness value per video
        huesat = random.uniform(0.5, 1.5)          # random hue & saturation value per video
        # Image Augmentation
        aug = iaa.Sequential([
            iaa.KeepSizeByResize(iaa.Crop(percent=0.15, keep_size=False)),
            iaa.LinearContrast(contrast),
            iaa.MultiplyBrightness(brightness),
            iaa.MultiplyHueAndSaturation(huesat, per_channel=False)
        ])      
        '''
        
        # Trial II
        translate_x = random.uniform(-0.2, 0.2)     # random x translation per video
        translate_y = random.uniform(-0.2, 0.2)     # random y translation per video
        # Image Augmentation
        aug = iaa.Sequential([
            iaa.Affine(translate_percent={"x": translate_x, "y": translate_y})
        ])

        '''
        # Trial III
        translate_x = random.uniform(-0.2, 0.2)     # random x translation per video
        translate_y = random.uniform(-0.2, 0.2)     # random y translation per video
        # Image Augmentation
        aug = iaa.Sequential([
            iaa.KeepSizeByResize(iaa.Crop(percent=0.15, keep_size=False)),
            iaa.Affine(translate_percent={"x": translate_x, "y": translate_y})
        ])
        '''

        if os.path.exists(os.path.join(folderOut, str(int(directory.name)+200000) +'.mp4')):
            continue
        for filename in sorted(glob.glob(directory.path + '\*.jpg')):
            img = cv2.imread(filename)
            # Do image augmentation
            img_aug = aug.augment_image(img)
            img_array.append(img_aug)

        height, width, layers = img_aug.shape
        size = (width,height)
        # Add 200000 to video's number to make the new filename for output augmented videos
        out = cv2.VideoWriter(os.path.join(folderOut, str(int(directory.name)+200000) +'.mp4'), fourcc, 12.0, size)

        for i in range(len(img_array)):
            out.write(img_array[i])     # Save video file
        out.release()

        if num > 0 and num % log_int == 0:
            vid_e_time = time.time()
            loop_t = vid_e_time - vid_s_time
            vid_s_time = time.time()
            t = (len(directories) - num)/log_int * loop_t /3600
            logger.info('%04d/%04d is done (%4.2f%%). Time for processing %02d videos: %4.2f seconds. ETA: %4.2fh' % (num, len(directories),num/len(directories)*100,log_int, loop_t, t))
