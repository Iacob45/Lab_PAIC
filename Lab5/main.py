import numpy as np
import matplotlib.pyplot as plt
import scipy
from skimage import io, color
from Core.utilities import show_image_rgb, show_image_gray, show_plot, apply_conv_filter, \
    lexicographic_histogram_equalization

IMAGE_PATH = '../Images/lena.png'


def main():
    I = io.imread(IMAGE_PATH)
    I = I[:, :, 0:3]
    Y = color.rgb2gray(I)

    show_image_gray(Y)

    # h, bins = np.histogram(Y, bins=256, range=(0, 1.0000000001))
    # hnormat = h / np.sum(h)

    # show_plot(hnormat, title='Histograma Y, cu hist, float64')

    # Ymedie = scipy.signal.convolve2d(Y, np.ones((7, 7)) / 49.0, 'same', 'symm')
    # h, bins = np.histogram(Ymedie, bins=256, range=(0, 1.0000000001))
    # hnormat = h / np.sum(h)

    # show_image_gray(Ymedie)
    # show_plot(hnormat, title='Histograma Ymedie, cu hist, float64')
    # show_image_gray(np.abs(Y-Ymedie), autoscale=True)

    # Yout = apply_conv_filter(Y, 31)
    # show_image_gray(Yout)
    # show_image_gray(np.abs(Y-Yout))

    Yout = lexicographic_histogram_equalization(Y)

    h, bins = np.histogram(Yout, bins=256, range=(0, 1.0000000001))
    hnormat = h / np.sum(h)

    show_image_gray(Yout)
    show_plot(hnormat, title='Histograma Ymedie, cu hist, float64')


if __name__ == '__main__':
    main()
    plt.show()
