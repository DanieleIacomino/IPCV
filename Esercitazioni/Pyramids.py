import imageio as io
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
from skimage import color
from skimage import transform

sigma=5
size=55

x,y=np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

L=(x**2+y**2-2*sigma**2)/(2*np.pi*sigma**6)*np.exp(-(x**2+y**2)/(2*sigma**2))

#plt.figure()
plt.imshow(L, cmap='gray')
plt.colorbar()
plt.show()

from matplotlib import cm

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(x, y, L, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_title("Laplacian of Gaussian (3D surface)")


plt.show()
def my_gaussian(sigma, size):
    x,y=np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    G = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return G

def my_laplacian(sigma, size):
    x,y=np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    L=(x**2+y**2-2*sigma**2)/(2*np.pi*sigma**6)*np.exp(-(x**2+y**2)/(2*sigma**2))

    return L



path =r'C:\Users\danie\Desktop\IPCV\Foto Esercitazioni'

Img=io.imread(os.path.join(path,'lighthouse.png'))

Img1=Img.copy()
Img1=Img1.astype(np.float32)/255
if len(Img1.shape)==3:
    Img1gray= color.rgb2gray(Img1)
Dimgauss=55
gauss_pyramid=[]

gauss_pyramid.append(Img1gray.copy())
for i in range(1,5):
    Gauss=my_gaussian(5,55)
    Smoothed=scipy.signal.convolve2d(gauss_pyramid[i-1],Gauss,mode='same',boundary='symm')
    subsampled=Smoothed[::2,::2]
    gauss_pyramid.append(subsampled)

#plt.figure(figsize=(10, 6))
for k in range(len(gauss_pyramid)):
    plt.subplot(2, (len(gauss_pyramid)+1)//2, k+1)
    plt.imshow(gauss_pyramid[k], cmap='gray')
    plt.axis('off')
    plt.title(f'Level {k}')
plt.suptitle('Gaussian Pyramid')
plt.tight_layout()
plt.show()




laplacian_pyramid=[]
I_lp=np.copy(Img1gray)
h_lap=55
for k in range(0,6):
    lfiltered=scipy.signal.convolve2d(I_lp,my_laplacian(5,55),mode='same')
    laplacian_pyramid.append(lfiltered)
    I_lp=I_lp[::2,::2]

#plt.figure()
for k in range(5):  # show first 5 levels
    plt.subplot(2, 3, k+1)
    plt.imshow(laplacian_pyramid[k], cmap='gray')
    plt.axis('off')
    plt.title(f'Level {k}')

plt.suptitle('Laplacian Pyramid')
plt.tight_layout()
plt.show()


diff_pyramid=[]
diff_pyramid.append(Img1gray.copy())

for k in range(len(gauss_pyramid)):
    G1=my_gaussian(5,55)
    G2=my_gaussian(3,55)
    Gfiltered1=scipy.signal.convolve2d(gauss_pyramid[k],G1,mode='same')
    Gfiltered2=scipy.signal.convolve2d(gauss_pyramid[k],G2,mode='same')
    diff=Gfiltered1-Gfiltered2
    diff_pyramid.append(diff)
    I_lp=I_lp[::2,::2]

#plt.figure()
for k in range(5):
    plt.subplot(2,3,k+1) 
    plt.imshow(diff_pyramid[k], cmap = 'gray')

plt.suptitle('DoG Pyramid')    
#plt.show()


I_rec = gauss_pyramid[-1]
levels = len(gauss_pyramid)
scale = np.arange(0, levels-1)   # Gaussian pyramid scales go from 0 to 5
scale = np.flip(scale)           # We need to start from the coarsest level and move backwards



plt.figure()
for k in scale:
    # Perform upsampling using bicubic interpolation (order=3).
    I_up = transform.rescale(I_rec, 2, order=3) 
    
    # Add the detail image (from the Laplacian pyramid) back to reconstruct.
    I_rec = I_up + laplacian_pyramid[k]
     
# Compare original vs reconstructed image
plt.figure()
plt.subplot(121), plt.imshow(Img1gray, cmap='gray'), plt.title('Original')
plt.subplot(122), plt.imshow(I_rec, cmap='gray'), plt.title('Reconstructed')
plt.show()