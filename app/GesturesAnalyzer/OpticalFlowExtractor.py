import cv2
import numpy as np
import time


class OpticalFlowExtractor:

    def __init__(self):
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.size = (170,100)
        self.folderOut = 'FLOW_OUT_PATH' # MUST BE CHANGED 'ie/

    def extract_optical_flow(self, video, _id):
        start = time.time()
        cap = cv2.VideoCapture(video)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ret, frame1 = cap.read()
        prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255

        array = []

        for i in range(length-1):
            ret, frame2 = cap.read()
            next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
            
            array.append(rgb)
            prvs = next

        cap.release()
        cv2.destroyAllWindows()

        flow_path = self.folderOut + "opfw_" + video.split('/')[-1]
        out = cv2.VideoWriter(flow_path, self.fourcc, 12.0, self.size)
            
        for i in range(len(array)):
            out.write(array[i])
        out.release()
        end = time.time()
        print("Extract optical flow:" + str(end-start))
        return flow_path
