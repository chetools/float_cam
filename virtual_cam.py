import pyvirtualcam
import numpy as np
import time

with pyvirtualcam.Camera(1280,720,30) as cam:
    while True:

        frame = np.zeros((cam.height, cam.width, 4), np.uint8) # RGBA
        frame[:,:,0]=255
        frame[:,:,1]=0
        frame[:,cam.width//2:cam.width,2] = cam.frames_sent % 255 # grayscale animation
        frame[:,:,3] = 255
        cam.send(frame)
        cam.