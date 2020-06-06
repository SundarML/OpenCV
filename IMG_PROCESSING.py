#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import matplotlib.pyplot as plt
import cv2


# In[12]:


img1 = cv2.imread('ai.png')
rows, cols, channel = img1.shape
img2gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

ret, mask = cv2.threshold(img2gray, 220, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('image',img2gray)
#print(rows, cols, channel)
#cv2.imshow('img1', img1)


cv2.waitKey(0)
cv2.destroyAllWindows()


# In[8]:


import cv2
import numpy as np
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_red = np.array([50,0,0])
    upper_red = np.array([255,255,255])
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    
    kernel = np.ones((15,15), np.float32)/255
    smoothed = cv2.filter2D(res, -1, kernel)
    
    blur = cv2.GaussianBlur(res, (15,15), 0)
    median = cv2.medianBlur(res, 15)
    
    cv2.imshow('blur', blur)
    cv2.imshow('median', median)
    cv2.imshow('smoothed', smoothed)
    #cv2.imshow('frame', frame)
    #cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    #cv2.imshow('hsv', hsv)
    #cv2.imshow()
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
        
cv2.destroyAllWindows()
cap.release()


# In[22]:


import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_red = np.array([150, 0, 0])
    upper_red = np.array([255,255,255])
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_or(frame, frame, mask=mask)
    
    kernel = np.ones((2,2), np.uint8)
    erosion = cv2.erode(mask, kernel, iterations=1)
    dilation = cv2.dilate(mask, kernel, iterations=1)
    
    cv2.imshow('mask',mask)
    #cv2.imshow('res',res)
    cv2.imshow('erosion', erosion)
    cv2.imshow('dilation', dilation)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
        
cv2.destroyAllWindows()
cap.release()
    


# In[4]:


import numpy as np
import cv2

img = cv2.VideoCapture(0)

while True:
    _, frame = img.read()
    laplacian = cv2.Laplacian(frame, cv2.CV_64F)
    
    edges = cv2.Canny(frame, 10,10)
    
    
    #cv2.imshow('laplacian', laplacian)
    cv2.imshow('edges', edges)
    
    k = cv2.waitKey(5)
    if k == 27:
        break
cv2.destroyAllWindows()
img.release()


# # Template Matching

# In[31]:


import numpy as np
import cv2

img_bgr = cv2.imread('window_1.jpg')
img_gry = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
template = cv2.imread('window_2.jpg', 0)
w, h = template.shape[::-1]


res = cv2.matchTemplate(img_gry, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.5
loc = np.where(res>=threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(img_bgr, pt, (pt[0]+w, pt[1]+h), (0, 255, 255), 2)
#print(loc)

cv2.imshow('detected', img_bgr)
#cv2.imshow('image',img_gry)
#cv2.imshow('template', template)


cv2.waitKey(0)
cv2.destroyAllWindows()


# In[47]:


import numpy as np
import cv2

img_bgr = cv2.imread('_1.jpg')
img_gry = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
template = cv2.imread('window_2.jpg', 0)
w, h = template.shape[::-1]

#print(w)
#print(h)
res = cv2.matchTemplate(img_gry, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where(res>=threshold)
print(res)
print(enumerate(loc))
#for pt in zip(*loc[::-1]):
#    cv2.rectangle(img_bgr, pt, (pt[0]+w, pt[1]+h), (0, 255, 255), 2)

#for pt in zip(*loc[::-1]):
#print(zip(*loc[::-1]))

#cv2.imshow('detected', img_bgr)
#cv2.imshow('image',img_gry)
#cv2.imshow('template', template)


cv2.waitKey(0)
cv2.destroyAllWindows()


# # Foreground Extraction

# In[ ]:





# # Feature Tracking

# In[6]:


import numpy as np
import cv2
img = cv2.imread('gw_1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

corners = cv2.goodFeaturesToTrack(gray, 500, 0.01, 10)
corners = np.int0(corners)

for corner in corners:
    x, y = corner.ravel()
    cv2.circle(img, (x,y), 3, 255, -1)
cv2.imshow('Corner', img)


cv2.waitKey(0)
cv2.destroyAllWindows()


# # Image Matching

# In[11]:


import numpy as np
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('gw_1.jpg',1)
img2 = cv2.imread('gw_2.jpg',1)

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key = lambda x: x.distance)

img3 = cv2.drawMatches(img1,kp1, img2,kp2, matches[:5], None, flags=2)
#plt.figure(figsize=(15,10))
#plt.imshow(img3)
#plt.show()

cv2.imshow('image',img3)

cv2.waitKey(0)
cv2.destroyAllWindows()


# # Foreground image capturing

# In[2]:


import numpy as np
import cv2

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    
    cv2.imshow('original', frame)
    cv2.imshow('fg', fgmask)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()    
cv2.destroyAllWindows()


# In[7]:


first = 'sundar'
last = 'rajan'
f"{first} {last}"
first+' '+last

