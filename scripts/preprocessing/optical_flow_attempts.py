import cv2
import numpy as np
import time
import logging
import os
import argparse
from sklearn.cluster import MiniBatchKMeans

def parse_args():
    parser = argparse.ArgumentParser(description='Extract optical flow.')
    parser.add_argument('--mode', type=str, default='mask',
                        help="Optical flow mode. Modes: standard, standardPrev, quantNoise, noise, luminosity, sparse, mask")
    parser.add_argument('--lumin', type=int, default=None,
                        help="For luminosity mode, bitwise right shift number")
    
    opt = parser.parse_args()
    return opt

def getRGBflow(flow, hsvVal): 
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsvVal[...,0] = ang*180/np.pi/2
    hsvVal[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsvVal,cv2.COLOR_HSV2BGR)
    return rgb

def getRGBflowMask(flow, hsvVal, mask): 
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsvVal[...,0] = ang*180/np.pi/2
    hsvVal[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsvVal,cv2.COLOR_HSV2BGR)
    masked = cv2.bitwise_and(rgb, rgb, mask=mask)
    return masked
 
def quantImage(rgbImage):
    (h, w) = rgbImage.shape[:2]
    image = cv2.cvtColor(rgbImage, cv2.COLOR_BGR2LAB)

    image = image.reshape((image.shape[0] * image.shape[1], 3))
    # apply k-means using the specified number of clusters and then create the quantized image based on the predictions
    clt = MiniBatchKMeans(3)
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]
    # reshape the feature vectors to images
    quant = quant.reshape((h, w, 3))
    image = image.reshape((h, w, 3))
    # convert from L*a*b* to RGB
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
    return quant

def reduceLight(image, lum):
    width, height, _ = image.shape
    im = np.zeros_like(image)
    for x in range(width):
        for y in range(height):
            px = image[x,y]
            im[x,y] = px >> lum
    return im

def standardOpFw(prs, frame_gray, flowPrev=None):
    flow = cv2.calcOpticalFlowFarneback(prvs, frame_gray, flowPrev, 0.5, 3, 15, 3, 5, 1.1, 0)
    return flow


def getSparseFlow(prvs, frame_gray, p0, mask, **lk_params):
    p1, st, err = cv2.calcOpticalFlowPyrLK(prvs, frame_gray, p0, None, **lk_params)
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
    for i,(new,old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        flow = cv2.line(mask, (int(a),int(b)),(int(c),int(d)), color[i].tolist(), 2)
    p0 = good_new.reshape(-1,1,2)
    return p0, flow


def getLaplaceMask(frame_gray):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))
    image = cv2.GaussianBlur(frame_gray, (7, 7), 0)
    image = np.uint8(np.absolute(cv2.Laplacian(image, cv2.CV_64F, 7)))
    image = cv2.GaussianBlur(image, (7, 7), 0)
    image = (image * (255 / image.max())).astype(np.uint8)
    #print(np.average(image))
    x = image.astype(np.float32) / 255
    image[image>np.average(image)] = 255
    image[image<np.average(image)] = 0 
    openc = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return openc


if __name__ == '__main__':
    ## INIT ##
    folderOut = "/Users/sofia/code/Dataset/optical_flow/"
    if not os.path.exists(folderOut):
        os.makedirs(folderOut)
    # init attrs
    args = parse_args()
    method = args.mode
    lum = args.lumin
    arrayOri = []
    array = []
    video = []
    videoQ = []
    flowPrev = None
    rgbFlow = None
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    ## For sparse optical flow ##
    feature_params = dict(maxCorners = 50,qualityLevel = 0.3,minDistance = 15, blockSize = 7) # Parameters for lucas kanade optical flow
    lk_params = dict(winSize  = (15,15),maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    color = np.random.randint(0,255,(100,3))    # Create some random colors
    ## End for sparse optical flow ##

    #Optical flow types init
    optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
    optical_flow_farne = cv2.FarnebackOpticalFlow_create()
    
   

    for vid, vline in enumerate(["/Users/sofia/62.mp4"]):
            
        video_path = vline.split()[0]
        video_name = video_path.split('/')[-1]

        cap = cv2.VideoCapture(video_path)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ret, frame1 = cap.read()
        prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
       
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255
        p0 = cv2.goodFeaturesToTrack(prvs, mask = None, **feature_params)
        mask = np.zeros_like(frame1)
        for i in range(length-1):
            ret, frame2 = cap.read()
            video.append(frame2)
            frame_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            if method == 'standard':
                flow = standardOpFw(prvs, frame_gray)
                prvs = frame_gray
                rgbFlow = getRGBflow(flow, hsv)

            if method == 'standardPrev':
                flow = standardOpFw(prvs, frame_gray, flowPrev)
                prvs = frame_gray
                flowPrev = flow
                rgbFlow = getRGBflow(flow, hsv)

            elif method == 'quantNoise':
                img = cv2.GaussianBlur(frame2, (7, 7), 0)
                img = quantImage(img)
                videoQ.append(img)
                frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                flow = standardOpFw(prvs, frame_gray)
                prvs = frame_gray
                rgbFlow = getRGBflow(flow, hsv)
            
            elif method == 'noise':
                blur = cv2.GaussianBlur(frame_gray, (7, 7), 0)
                flow = standardOpFw(prvs, frame_gray)
                prvs = frame_gray
                flowPrev = flow
                rgbFlow = getRGBflow(flow, hsv)

            elif method == 'luminosity':
                frame = reduceLight(frame2,lum)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                flow = standardOpFw(prvs, frame)
                prvs = frame_gray
                rgbFlow = getRGBflow(flow, hsv)
                
            elif method == 'sparse':
                p0, rgbFlow = getSparseFlow(prvs, frame_gray, p0, mask, **lk_params)
                prvs = frame_gray

            elif method == 'mask':
                mask = getLaplaceMask(frame_gray)
                flow = standardOpFw(prvs, frame_gray)
                rgbFlow = getRGBflowMask(flow, hsv, mask)


            array.append(rgbFlow.copy())

        cap.release()
        cv2.destroyAllWindows()
        # cv2.imshow('im2', array[0])
        # cv2.waitKey(0)
        height, width, layers = rgbFlow.shape
        # size = (width,height)
        size = (width*3,height)

        out = cv2.VideoWriter(folderOut + "opfw_" + method  + "_" + video_name, fourcc, 12.0, size)
            
        for i in range(len(array)):
            out.write(np.concatenate((video[i], videoQ[i], array[i]), axis=1))
        out.release()



