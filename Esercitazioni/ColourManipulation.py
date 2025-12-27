# Import libraries
import os
import matplotlib.pyplot as plt
import imageio.v3 as io
import skimage.color as colour
import numpy as np  

path =r'C:\Users\danie\Desktop\IPCV\Foto Esercitazioni'
Img=io.imread(os.path.join(path,'lighthouse.png'))



Img=Img.astype(np.float32)/255
Img_hsv=colour.rgb2hsv(Img)

plt.figure()
plt.subplot(1,2,1),plt.imshow(Img),plt.title('RGB')
plt.subplot(1,2,2),plt.imshow(Img_hsv),plt.title('HSV')
plt.show()

Img_copy=Img.copy()
Img_copy[:,:,0]=0
plt.figure(),plt.imshow(Img_copy),plt.title('RGB - Red channel removed')
plt.show()

# S max
Img_copy2= Img_hsv.copy()
Img_copy2[:,:,1]=1
Img_copy2[:,:,2]=1
Img_copy2=colour.hsv2rgb(Img_copy2)
#V max
img_copy3=Img_hsv.copy()
img_copy3[:,:,0]=1
img_copy3[:,:,2]=1
img_copy3=colour.hsv2rgb(img_copy3)
#H max
img_copy4= Img_hsv.copy()
img_copy4[:,:,0]=1
img_copy4[:,:,1]=0
img_copy4=colour.hsv2rgb(img_copy4)

plt.figure()
plt.subplot(1,3,1)
plt.imshow(Img_copy2)
plt.axis('off')
plt.title('Hue')
plt.subplot(1,3,2)
plt.imshow(img_copy3)
plt.axis('off')
plt.title('Saturation')
plt.subplot(1,3,3)
plt.imshow(img_copy4)
plt.axis('off')
plt.title('Value')
plt.show()


Img_copy5=Img.copy()
H,W = Img_copy5.shape[:2]
h2, w2 = H//2, W//2
P1 = Img_copy5[0:h2,     0:w2    ]   # top-left
P2 = Img_copy5[0:h2,     w2:W    ]   # top-right
P3 = Img_copy5[h2:H,     0:w2    ]   # bottom-left
P4 = Img_copy5[h2:H,     w2:W    ]   # bottom-right
P1[:,:,0]=0

patch2 = P2.copy()
patch2_hsv = colour.rgb2hsv(patch2)
patch2_hsv[..., 1] = 1.0    # saturazione = 1
patch2_rgb = colour.hsv2rgb(patch2_hsv)
Img_copy5[0:h2, w2:W] = patch2_rgb   # scrivi indietro

# Patch 3: swap canali (RGB → GBR)
patch3 = P3.copy()
patch3_swapped = patch3[..., [1,2,0]]
Img_copy5[h2:H, 0:w2] = patch3_swapped


plt.imshow(Img_copy5)              # float [0,1] → ok
plt.axis('off')
plt.title("Quadrants edited")
plt.show()

