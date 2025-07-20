import numpy as np
import cv2

def deskew(img):
    m=cv2.moments(img)
    if abs(m['mu02'])<1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*28*skew], [0,1,0]])
    return cv2.warpAffine(img, M, (28,28),
                          flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)

def preprocess(img):
    img=deskew(img)
    s, img=cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img

def batch_preprocess(images):
    return np.stack([preprocess(im) for im in images], axis=0)
