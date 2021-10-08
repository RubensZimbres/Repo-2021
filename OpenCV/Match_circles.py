import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('/home/anaconda3/lib/python3.7/site-packages/tensorflow/models/research/object_detection/tabela_circle2.jpg', 0)

template = cv2.imread('/home/anaconda3/lib/python3.7/site-packages/tensorflow/models/research/object_detection/circle2.jpg', 0)
h, w = template.shape

res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

cv2.imwrite('res.png',img)
