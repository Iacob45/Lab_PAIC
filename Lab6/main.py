from matplotlib import pyplot as plt
import numpy as np
import scipy
from skimage import io, color

from Core.constants import *
from Core.utilities import *


def main():
    I = io.imread(IMAGE_PATH)
    I = I[:, :, 0:3]

    zg = np.random.normal(0, 30, (100, 100))

    color1 = 75*np.ones((100, 50))
    color2 = 150*np.ones((100, 50))
    img = np.hstack((color2, color1))

    testzg = img + zg

    clipped_img = np.clip(testzg, 0, 255)

    print(MSE(img, clipped_img))

    results1 = []
    plt.figure(figsize=(8, 4))

    for i in range(1, 10):
        kernel = np.ones((2*i+1, 2*i+1))/(2*i+1)**2
        rez_medie = scipy.signal.convolve2d(clipped_img, kernel, 'same', 'symm')
        results1.append(MSE(img, rez_medie))
        show_plot(rez_medie[50, :], reuse_figure=True)

    results2 = []
    for i in range(1, 10):
        zg = np.random.normal(0, 2**i, np.shape(img))
        testzg = img + zg
        testzg = np.clip(testzg, 0, 255)
        results2.append(MSE(img, testzg))

    plt.figure(figsize=(8, 4))
    show_plot(results1, reuse_figure=True)
    show_plot(results2, reuse_figure=True)


if __name__ == '__main__':
    main()
    plt.show()
