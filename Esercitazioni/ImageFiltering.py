# Import libraries
import os
import matplotlib.pyplot as plt
import imageio.v3 as io
import skimage.color as color
import numpy as np  
from scipy import ndimage
from scipy.signal import convolve2d
path =r'C:\Users\danie\Desktop\IPCV\Foto Esercitazioni'

Img=io.imread(os.path.join(path,'lighthouse.png'))

Img1=Img.copy()
Img1=Img1.astype(np.float32)/255
if len(Img1.shape)==3:
    Img1gray= color.rgb2gray(Img1)

plt.figure()
plt.subplot(1,2,1),plt.imshow(Img1)
plt.subplot(1,2,2),plt.imshow(Img1gray,cmap='gray')
plt.show()


# --- Define Boxcar Kernel Manually ---
side_box = 33
h_box = np.ones((side_box, side_box), dtype=float)
h_box = h_box / np.sum(h_box)


# --- Define Gaussian Kernel Manually ---
sigma = 5 # Standard Deviation
side_gauss = 55 # Kernel Size (55x55)
[x, y] = np.meshgrid(
    np.arange(-side_gauss//2, side_gauss//2 + 1),
    np.arange(-side_gauss//2, side_gauss//2 + 1)
)
h_gauss = np.exp(-(x**2 + y**2) / (2 * sigma**2))
h_gauss = h_gauss / np.sum(h_gauss)
# --- Boxcar impulse response Kernel Manually ---
temp = np.zeros((side_box, side_box))
temp[side_box // 2, side_box // 2] = 1
H_box = ndimage.uniform_filter(temp, side_box)


# --- Define Gaussian Kernel via Impulse Response using ndimage ---
temp2 = np.zeros((side_gauss, side_gauss))
temp2[side_gauss // 2, side_gauss // 2] = 1
H_gauss = ndimage.gaussian_filter(temp2, sigma=sigma, mode='constant', cval=0.0)


# --- Visualize Boxcar Kernels ---


plt.figure()
plt.subplot(121),plt.imshow(h_box,cmap='gray',vmin=-1/side_box**2, vmax=2/side_box**2)
plt.subplot(122),plt.imshow(H_box,cmap='gray',vmin=-1/side_box**2, vmax=2/side_box**2)
plt.show()
# --- Visualize Gaussian Kernels --
plt.figure()
plt.subplot(121),plt.imshow(h_gauss,cmap='gray')
plt.subplot(122),plt.imshow(H_gauss,cmap='gray')
plt.show()


def Box_car(size):
    side_box = size
    h_box = np.ones((side_box, side_box), dtype=float)
    h_box = h_box / np.sum(h_box)
    return h_box

boxcartest= Box_car(10)
print(boxcartest)
print('test di normalizzazione', boxcartest.sum())


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


kernel_gauss = Gauss(55, 5)
print('Somma del kernel:', kernel_gauss.sum())

plt.imshow(kernel_gauss, cmap='gray')
plt.title('Gaussian Kernel')
plt.colorbar()
plt.show()

side_box = 11
side_gauss = 21
sigma=3


hbox=Box_car(side_box)
h_gauss=Gauss(side_gauss,sigma)

Filteredbox=convolve2d(Img1gray,h_box,mode='same',boundary='symm')
FilteredGauss=convolve2d(Img1gray,h_gauss,mode='same',boundary='symm')

plt.figure(figsize=(12, 6))
plt.subplot(131), plt.imshow(Img1gray, cmap='gray', vmin=0, vmax=1), plt.title('Original'), plt.axis('off')
plt.subplot(132), plt.imshow(Filteredbox, cmap='gray', vmin=0, vmax=1), plt.title('Box Filter (Manual Kernel)'), plt.axis('off')
plt.subplot(133), plt.imshow(FilteredGauss, cmap='gray', vmin=0, vmax=1), plt.title('Gaussian Filter (Manual Kernel)'), plt.axis('off')
plt.show()


side_box   = 11     # finestra per uniform_filter
sigma      = 3      # deviazione standard per gaussian_filter
# Per rendere il Gauss "simile" a un kernel di lato prefissato:
# side_gauss = 21
# truncate = ((side_gauss - 1) / 2) / sigma

# --- Built-in ndimage filters ---
I_box_nd   = ndimage.uniform_filter(Img1gray, size=side_box, mode='reflect')
I_gauss_nd = ndimage.gaussian_filter(Img1gray, sigma=sigma, mode='reflect')  # oppure aggiungi truncate=...

plt.figure(figsize=(12, 6))
plt.subplot(131), plt.imshow(Img1gray, cmap='gray', vmin=0, vmax=1),       plt.title('Original'),                   plt.axis('off')
plt.subplot(132), plt.imshow(I_box_nd, cmap='gray', vmin=0, vmax=1), plt.title('Box (ndimage.uniform_filter)'), plt.axis('off')
plt.subplot(133), plt.imshow(I_gauss_nd, cmap='gray', vmin=0, vmax=1), plt.title('Gaussian (ndimage.gaussian_filter)'), plt.axis('off')
plt.show()
