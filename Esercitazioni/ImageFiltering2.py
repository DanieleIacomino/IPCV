# Import libraries
import os
import matplotlib.pyplot as plt
import imageio.v3 as io
import skimage.color as color
import numpy as np  
from scipy import ndimage
import cv2 as cv
from scipy.signal import convolve2d
path =r'C:\Users\danie\Desktop\IPCV\Foto Esercitazioni'

Img=io.imread(os.path.join(path,'lighthouse.png'))

Img1=Img.copy()
Img1=Img1.astype(np.float32)/255
if len(Img1.shape)==3:
    Img1gray= color.rgb2gray(Img1)

def Gauss(size, sigma):
    """
    Create a 2D Gaussian kernel with given size and standard deviation sigma.
    size: odd integer (e.g., 55)
    sigma: standard deviation of the Gaussian
    """
    # coordinate grid
    x, y = np.meshgrid(
        np.arange(-size//2, size//2 + 1),
        np.arange(-size//2, size//2 + 1)
    )

    # Gaussian formula
    h_gauss = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    h_gauss = h_gauss / np.sum(h_gauss)  # normalize

    return h_gauss


def Box_car(size):
    side_box = size
    h_box = np.ones((side_box, side_box), dtype=float)
    h_box = h_box / np.sum(h_box)
    return h_box

#Sobel
derX=np.array([[-1,0,1],
              [-2,0,2],
              [-1,0,1]],dtype=float)

derY=np.array([[-1,-2,-1],
               [0,0,0],
               [1,2,1]],dtype=float)


def gauss_derivative_kernels(size, sigma):
    assert size % 2 == 1, "side_gauss dev'essere dispari"
    r = size // 2
    # indexing='ij' -> asse 0 = y (righe), asse 1 = x (colonne)
    y, x = np.meshgrid(np.arange(-r, r+1), np.arange(-r, r+1), indexing='ij')
    G = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    # la derivata di G (non si normalizza: la somma è ~0 per definizione)
    Gx = -(x / (sigma**2)) * G   # ∂G/∂x
    Gy = -(y / (sigma**2)) * G   # ∂G/∂y
    return Gx, Gy



Sx=convolve2d(Img1gray,derX)
Sx=abs(Sx)
Sy=convolve2d(Img1gray,derY)
Sy=abs(Sy)

# Magnitudo del gradiente
G = np.hypot(Sx, Sy)
G = G / (G.max() + 1e-8)  # normalizza per un display pulito

plt.figure(figsize=(12, 8))
plt.subplot(2,2,1); plt.imshow(Img1gray, cmap='gray'); plt.title('Originale'); plt.axis('off')
plt.subplot(2,2,2); plt.imshow(Sx, cmap='gray');   plt.title('|∂I/∂x| (Sobel)'); plt.axis('off')
plt.subplot(2,2,3); plt.imshow(Sy, cmap='gray');   plt.title('|∂I/∂y| (Sobel)'); plt.axis('off')
plt.subplot(2,2,4); plt.imshow(G, cmap='gray');        plt.title('Magnitudo |∇I|');  plt.axis('off')
plt.tight_layout(); plt.show()


side_gauss=33
sigmas = [2,3,5,7]
dog_x = [] 
dog_y = []

for sig in sigmas:
    Gx, Gy = gauss_derivative_kernels(side_gauss, sig)
    Sx = convolve2d(Img1gray, Gx, mode='same', boundary='symm')
    Sy = convolve2d(Img1gray, Gy, mode='same', boundary='symm')
    dog_x.append(Sx)
    dog_y.append(Sy)



    # --- Plot comparativo: |∂I/∂x| e |∂I/∂y| per ciascun sigma ---
n = len(sigmas)
plt.figure(figsize=(3*n, 6))
for i, sig in enumerate(sigmas):
    Sx_i = dog_x[i]; Sy_i = dog_y[i]
    plt.subplot(2, n, i+1)
    plt.imshow(np.abs(Sx_i), cmap='gray')
    plt.title(f'|dI/dx| σ={sig}')
    plt.axis('off')

    plt.subplot(2, n, n+i+1)
    plt.imshow(np.abs(Sy_i), cmap='gray')
    plt.title(f'|dI/dy| σ={sig}')
    plt.axis('off')

plt.tight_layout()
plt.show()


edges = cv.Canny((Img1gray*255).astype(np.uint8), 100, 200)
plt.figure(figsize=(8, 6))
plt.subplot(1,2,1); plt.imshow(Img1gray, cmap='gray'); plt.title('Originale'); plt.axis('off')  
plt.subplot(1,2,2); plt.imshow(edges, cmap='gray'); plt.title('Bordi (Canny)'); plt.axis('off')
plt.tight_layout(); plt.show()
watershed = cv.watershed((Img1gray*255).astype(np.uint8), markers=None)
plt.figure(figsize=(8, 6)) 
plt.subplot(1,2,1); plt.imshow(Img1gray, cmap='gray'); plt.title('Originale'); plt.axis('off')
plt.subplot(1,2,2); plt.imshow(watershed, cmap='gray'); plt.title('Watershed'); plt.axis('off')
plt.tight_layout(); plt.show()