import numpy as np
import matplotlib.pyplot as plt
import scipy
from skimage import io, color

from Core.utilities import show_image_rgb, show_image_gray, show_plot, T, expandare_liniara, prelucrare_scalara, HE, \
    HE_with_window, CLHE

IMAGE_PATH = '../Images/uneven-face-mic.png'


# -------------------------------------------------------
# -------------------------------------------------------

I = io.imread(IMAGE_PATH)
I = I[:, :, 0:3]

Y = color.rgb2gray(I)

Ystricat = Y/10 + 0.6

# -------------------------------------------------------

# show_image_gray(Ystricat)

Ycorectat = prelucrare_scalara(Ystricat)

# show_image_gray(Ycorectat)

# -------------------------------------------------------

h, bins = np.histogram(Ystricat, bins=256, range=(0, 1.0000000001))
hnormat = h/np.sum(h)


# show_plot(hnormat, title='Histograma Ystricat, cu hist, float64')

# -------------------------------------------------------

YY = np.uint8(255*Y)
hYY, bins = np.histogram(YY, bins=256, range=(0, 256))
hYY = hYY/np.sum(hYY)

# show_plot(hYY, title='Histograma Y, cu hist, uint8')

# -------------------------------------------------------

contor = np.zeros(256)
[L, C] = np.shape(YY)
for l in range(0, L):
    for c in range(0, C):
        contor[YY[l, c]] += 1

contor = contor/np.sum(contor)

# show_plot(contor, title='Histograma Y, cu vector de aparitie, uint8')

# -------------------------------------------------------

# show_plot(np.cumsum(hYY), title='Histograma Y cumulativa, cu hist, uint8')

# -------------------------------------------------------

rezultat = HE(Ystricat)

# show_image_gray(rezultat)

# -------------------------------------------------------

stanga = HE(Ystricat[:, :225])
dreapta = HE(Ystricat[:, 225:])
rezultat = np.concatenate((stanga, dreapta), axis=1)

# show_image_gray(rezultat)

# -------------------------------------------------------

Iout = np.zeros(np.shape(I))

Iout[:, :, 0] = HE(I[:, :, 0])
Iout[:, :, 1] = HE(I[:, :, 1])
Iout[:, :, 2] = HE(I[:, :, 2])

w = 5

hsv = color.rgb2hsv(I)
# hsv[:, :, 2] = HE_with_window(hsv[:, :, 2], w)
rgb = color.hsv2rgb(hsv)

# Y2 = HE_with_window(Y, w)

# show_image_rgb(rgb)

# -------------------------------------------------------

Y3, hout = CLHE(Y, 0.02)
show_image_gray(Y)
show_image_gray(Y3)

h, bins = np.histogram(Y3, bins=256, range=(0, 1.0000000001))
hnormat = h/np.sum(h)


show_plot(hnormat, title='Histograma Y3')

h, bins = np.histogram(Y, bins=256, range=(0, 1.0000000001))
hnormat = h/np.sum(h)


show_plot(hnormat, title='Histograma Y')

show_plot(hout, title='Histograma hout')
