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

def foto(Img_name):
    path =r'C:\Users\danie\Desktop\IPCV\Foto Esercitazioni'

    Img=io.imread(os.path.join(path,Img_name))
    
    return Img

def Img_normalize(I):
    I=I.astype(np.float32)/255
    return I

def my_laplacian(sigma, size):
    x,y=np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    L=(x**2+y**2-2*sigma**2)/(2*np.pi*sigma**6)*np.exp(-(x**2+y**2)/(2*sigma**2))


def laplacian_devirative_kernels(size, sigma):
    assert size % 2 == 1, "side_gauss dev'essere dispari"
    r = size // 2
    # indexing='ij' -> asse 0 = y (righe), asse 1 = x (colonne)
    y, x = np.meshgrid(np.arange(-r, r+1), np.arange(-r, r+1), indexing='ij')
    G = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    # la derivata di G (non si normalizza: la somma è ~0 per definizione)
    L = ((x**2 + y**2 - 2*sigma**2) / (sigma**4)) * G   # ∂²G/∂x² + ∂²G/∂y²
    return L

def Additive_gaussian_noise(I,sigma):
    noise_sigma = sigma
    noise = noise_sigma * np.random.randn(*I.shape)
    I_noisy = I.copy() + noise
    return I_noisy
def Box_car(size):
    side_box = size
    h_box = np.ones((side_box, side_box), dtype=float)
    h_box = h_box / np.sum(h_box)
    return h_box
def Gauss_kernel(size, sigma):
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

derY=np.array([[-1,-2,-1],
               [0,0,0],
               [1,2,1]],dtype=float)

def Gauss_derivative_kernels(size, sigma):
    assert size % 2 == 1, "side_gauss dev'essere dispari"
    r = size // 2
    # indexing='ij' -> asse 0 = y (righe), asse 1 = x (colonne)
    y, x = np.meshgrid(np.arange(-r, r+1), np.arange(-r, r+1), indexing='ij')
    G = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    # la derivata di G (non si normalizza: la somma è ~0 per definizione)
    Gx = -(x / (sigma**2)) * G   # ∂G/∂x
    Gy = -(y / (sigma**2)) * G   # ∂G/∂y
    return Gx, Gy


derX=np.array([[-1,0,1],
              [-2,0,2],
              [-1,0,1]],dtype=float)

def my_gaussian(sigma, size):
    x,y=np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    G = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return G

