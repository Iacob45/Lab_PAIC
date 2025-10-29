import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt

image1 = io.imread('../Images/BlackColourWhite.png')


def citire_black_white():
    RGB = image1[:, :, :3]

    RGB = prelucrare_color_in_hsv(RGB)

    show_image_rgb(RGB)


def show_image_gray(image):
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.colorbar()
    plt.show()


def show_image_rgb(image):
    print(np.shape(image))
    print(type(image[0, 0, 0]))

    plt.figure()
    plt.imshow(image)
    plt.colorbar()
    plt.show()


def show_image_hsv(image, plane):
    print(np.shape(image))
    print(type(image[0, 0, 0]))

    color_map = "gray"
    if plane == 0:
        color_map = "hsv"

    plt.figure()

    plt.imshow(image[:, :, plane], cmap=color_map, vmin=0, vmax=1)
    plt.colorbar()
    plt.show()


def alter_image_plane(image, plane, modifier_type, modifier):
    new_image = image.copy()
    if modifier_type:
        new_image[:, :, plane] = new_image[:, :, plane] * modifier
    else:
        new_image[:, :, plane] = new_image[:, :, plane] + modifier
    return new_image


def prelucrare_scalara(plane):
    plane = plane * 0.5
    return plane


def prelucrare_color_marginala(image):
    R = prelucrare_scalara(image[:, :, 0])
    G = prelucrare_scalara(image[:, :, 1])
    B = prelucrare_scalara(image[:, :, 2])

    new_image = np.zeros(np.shape(image))
    new_image[:, :, 0] = R
    new_image[:, :, 1] = G
    new_image[:, :, 2] = B

    return new_image


def prelucrare_color_in_hsv(image):
    HSV = color.rgb2hsv(image)

    HSV[:, :, 2] = prelucrare_scalara(HSV[:, :, 2])

    color.hsv2rgb(HSV)

    new_image = color.hsv2rgb(HSV)

    return new_image


citire_black_white()
