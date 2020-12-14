# Importing required libraries
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Loading the image 
img = cv2.imread('/Users/gongsha/Documents/Learning/AI/Pytorch/PCA/images/lena.png') #you can use any image you want.
plt.imshow(img)
plt.show()

# Splitting the image in R,G,B arrays.
 
blue,green,red = cv2.split(img)

#Applying to red channel and then applying inverse transform to transformed array.
def compressAndShowImage(primaryColorMatrix, totalSize, gridSize, componentCount):
    pca = PCA(n_components=componentCount)

    flat_primary_color = np.ravel(primaryColorMatrix)
    size = int(totalSize / gridSize)
    reshaped_color = np.reshape(flat_primary_color, (size, -1))
    color_transformed = pca.fit_transform(reshaped_color)
    reshaped_color_inverted = pca.inverse_transform(color_transformed)
    color_inverted = np.reshape(reshaped_color_inverted, (int(math.sqrt(totalSize)), -1))
    img_compressed = (np.dstack((color_inverted, color_inverted, color_inverted))).astype(np.uint8)
    plt.imshow(img_compressed)
    plt.show()

totalSize = 1024*1024

compressAndShowImage(red, totalSize, 64, 4)
compressAndShowImage(red, totalSize, 16, 1)