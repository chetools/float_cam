import glob
from pathlib import Path
import numpy as np
import cv2
import re

pat=re.compile(r'(\d+)')

files = glob.glob('background/background*.png')
num = np.argsort(np.array([int(pat.search(afile)[1]) for afile in files]))

background_images = np.empty((len(files),720,1280,3))

for i,afile in enumerate(files):
    background_images[num[i]]=cv2.imread(afile)