# Import libraries
import os
import matplotlib.pyplot as plt
import imageio.v3 as io
import skimage.color as sk
import numpy as np  

path =r'C:\Users\danie\Desktop\IPCV\Foto Esercitazioni'

Img=io.imread(os.path.join(path,'lighthouse.png'))

plt.imshow(Img)
plt.axis('off')
plt.show()

# Convert to grayscale
Img_gray=sk.rgb2gray(Img)
plt.imshow(Img_gray,cmap='gray')
plt.axis('off')
plt.show()

Img2 = io.imread(os.path.join(path,'cameraman.tif'))
Img2_gray=Img2.astype('float32')/255
plt.imshow(Img2_gray,cmap='gray')
plt.axis('off')
plt.show()
print(Img2_gray.shape)

peppers=io.imread(os.path.join(path,'peppers.png'))
plt.imshow(peppers)
plt.axis('off')
plt.show()

peppers_gray=peppers.astype('float32')/255

red= peppers_gray[:,:,0]
green= peppers_gray[:,:,1]  
blue= peppers_gray[:,:,2]


plt.figure()
plt.subplot(1,3,1), plt.imshow(red,cmap='gray',vmin=0,vmax=1), plt.title('red')
plt.subplot(1,3,2), plt.imshow(green,cmap='gray',vmin=0,vmax=1), plt.title('green')
plt.subplot(1,3,3), plt.imshow(blue,cmap='gray',vmin=0,vmax=1), plt.title('blue')

plt.show()

peppers_patch = peppers[100:300,200:300,:]
plt.figure()
plt.imshow(peppers_patch)
plt.axis('off')
plt.show()
print(peppers_patch.shape)