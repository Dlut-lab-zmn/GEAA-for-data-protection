
import numpy as np
import cv2
from pylab import *



font=cv2.FONT_HERSHEY_SIMPLEX
im=np.zeros((32,32),np.uint8)
img=cv2.putText(255-im,'F',(8,24),font,0.8,(0,0,0),2)
cv2.imwrite('6.png',img)