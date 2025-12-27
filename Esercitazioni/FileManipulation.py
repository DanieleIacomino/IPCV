# Import libraries
import os
import matplotlib.pyplot as plt
import imageio.v3 as io
import skimage.color as colour
import numpy as np  

path =r'C:\Users\danie\Desktop\IPCV\Foto Esercitazioni'

Img=io.imread(os.path.join(path,'lighthouse.png'))
Img_numpy=np.array(Img)
print(type(Img_numpy))
print(Img_numpy.shape)
np.save(os.path.join(path,'lighthouse_numpy.npy'),Img_numpy)


# leggi immagine di partenza
img = io.imread(os.path.join(path, "lighthouse.png"))

# salva in vari formati nella stessa cartella
io.imwrite(os.path.join(path, "lighthouse_out.png"), img)
io.imwrite(os.path.join(path, "lighthouse_out.jpg"), img[:, :, :3])  # tolgo alpha se c'è
io.imwrite(os.path.join(path, "lighthouse_out.bmp"), img)
io.imwrite(os.path.join(path, "lighthouse_out.tiff"), img)
io.imwrite(os.path.join(path, "lighthouse_out.webp"), img)

print("Immagini salvate in:", path)