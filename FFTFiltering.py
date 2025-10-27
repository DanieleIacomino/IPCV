# Import libraries
import os
import matplotlib.pyplot as plt
import imageio.v3 as io
import skimage.color as color
import numpy as np  
from scipy import ndimage
from scipy.signal import convolve2d
import Mylibrary as ml
derX=np.array([[-1,0,1],
              [-2,0,2],
              [-1,0,1]],dtype=float)
path =r'C:\Users\danie\Desktop\IPCV\Foto Esercitazioni'

Img=io.imread(os.path.join(path,'lighthouse.png'))

Img1=Img.copy()
Img1=Img1.astype(np.float32)/255
if len(Img1.shape)==3:
    Img1gray= color.rgb2gray(Img1)

Ifft=np.fft.fft2(Img1gray)

Ifft=np.fft.fftshift(Ifft)


#plt.figure(figsize=(12, 6))
#plt.subplot(1,2,1) 
#plt.imshow(Img1gray, cmap ='gray'), plt.title('Original')
#plt.subplot(1,2,2) 
#plt.imshow(np.log(np.abs(Ifft)), cmap ='bwr'), plt.title('Fourier Transform')
#plt.show()


# Create a low-pass filter (example) and demonstrate proper convolution use
# NOTE: convolve2d operates on spatial-domain (real) images. `Ifft` is complex
# (frequency-domain). Applying convolve2d to the FFT array is incorrect.
# If you want the Sobel/derivative result, convolve the spatial image instead:
rows, cols = Img1gray.shape

# Spatial-domain convolution (Sobel X)
Img_sobel_x = convolve2d(Img1gray, derX, mode='same', boundary='symm')
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(Img_sobel_x, cmap='gray')
plt.title('Sobel X (spatial convolution)')
plt.axis('off')

# Frequency-domain equivalent (demonstration): multiply FFTs instead of convolving
# Build a padded kernel the same size as the image, take its FFT, multiply and inverse-FFT
kernel_padded = np.zeros_like(Img1gray)
kh, kw = derX.shape
kernel_padded[:kh, :kw] = derX
H = np.fft.fft2(kernel_padded)

# recompute unshifted FFT of the image (we'll use non-shifted FFT for multiplication)
Ifft_noshift = np.fft.fft2(Img1gray)
filtered_freq = np.fft.ifft2(Ifft_noshift * H)

plt.subplot(1,2,2)
plt.imshow(np.real(filtered_freq), cmap='gray')
plt.title('Sobel X (frequency-domain multiplication)')
plt.axis('off')
plt.tight_layout()
plt.show()