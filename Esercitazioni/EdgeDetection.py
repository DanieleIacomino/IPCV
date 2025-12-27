# Import libraries
import os
import matplotlib.pyplot as plt
import imageio.v3 as io
import skimage.color as sk
import numpy as np  
from skimage import color
from scipy import ndimage
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.filters import threshold_otsu
from scipy.signal import convolve2d

def my_gaussian(sigma, size):
    x,y=np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    G = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return G


path =r'C:\Users\danie\Desktop\IPCV\Foto Esercitazioni'

Img_gray=io.imread(os.path.join(path,'coins.png'))
Img_gray=Img_gray.astype(np.float32)/255
def my_laplacian(sigma, size):
    x,y=np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    L=(x**2+y**2-2*sigma**2)/(2*np.pi*sigma**6)*np.exp(-(x**2+y**2)/(2*sigma**2))

    return L

derX=np.array([[-1,0,1],
              [-2,0,2],
              [-1,0,1]],dtype=float)
from skimage.filters import threshold_otsu

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

Dx=ndimage.convolve(Img_gray,derX,mode='reflect')
Dy=ndimage.convolve(Img_gray,derY,mode='reflect')
tg=0.5

Edges=np.sqrt(Dx**2+Dy**2)>tg


plt.figure()
plt.title("Edges via Gradient Magnitude")
plt.imshow(Edges, cmap='gray')
plt.axis('off')
plt.show()

IMG2=Img_gray.copy()

sigma =3
size = 33
tL = 0.00005
L = my_laplacian(sigma, size)
I_lap=convolve2d(IMG2,L,mode='same',boundary='symm')
zero_cross = (I_lap * ndimage.shift(I_lap, (0,1)) < 0)  # Zero crossing
strong_enough = np.abs(I_lap) > tL # Thresholding
edges_L = zero_cross & strong_enough

plt.figure()
plt.title("Edges via Laplacian")
plt.imshow(edges_L, cmap='gray')
plt.axis('off')
plt.show()


from skimage import feature

# --- Apply Canny edge detector to clean image ---
E_clean = feature.canny(IMG2, sigma=1, low_threshold=0.1, high_threshold=0.3)

plt.figure(figsize=(12,4))
plt.title('Canny Algorithm')
plt.imshow(E_clean, cmap = 'gray')
plt.axis('off')


# --- Inject Gaussian noise ---
noise_sigma = 0.05
noise = noise_sigma * np.random.randn(*Img_gray.shape)
I_noisy = Img_gray.copy() + noise

# Visualize original, noise, and noisy image
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(Img_gray, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(noise, cmap='gray')
plt.title("Gaussian Noise (σ = 5)")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(I_noisy, cmap='gray')
plt.title("Noisy Image")
plt.axis('off')
plt.tight_layout()
plt.show()

# --- Apply Canny to noisy image ---
E_noisy = feature.canny(I_noisy, sigma=1, low_threshold=0.1, high_threshold=0.3)

# --- Compare edges ---
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(E_clean, cmap='gray')
plt.title("Edges - Clean Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(E_noisy, cmap='gray')
plt.title("Edges - Noisy Image")
plt.axis('off')
plt.tight_layout()
plt.show()


from skimage.segmentation import watershed
Img_smoot=convolve2d(Img_gray, my_gaussian(5,33), mode='same', boundary='symm')
tg=0.5
Img_thresholded=Img_smoot>tg
Dem=ndimage.distance_transform_edt(Img_thresholded)
Markers=ndimage.label(Img_thresholded)[0]
Labels=watershed(-Dem,Markers,mask=Img_thresholded)

plt.figure()
plt.imshow(Labels, cmap='gray')
plt.title("Watershed Segmentation")
plt.axis('off')
plt.show()