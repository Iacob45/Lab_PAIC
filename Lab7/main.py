from matplotlib import pyplot as plt
from skimage import io

from Core.constants import IMAGE_PATH
from Core.denoise import median_marginal, median_vectorial, median_WMF, filtru_liniar_adaptiv_distanta_MM
from Core.noise import pune_zgomot_impulsiv
from Core.utilities import show_image_rgb


def main():
    I = io.imread(IMAGE_PATH)
    I = I[:, :, :3]

    show_image_rgb(I)
    noised_image = pune_zgomot_impulsiv(I, 0.1, 0.1)
    denoised_image1 = median_marginal(noised_image, 1)
    # denoised_image2 = median_vectorial(noised_image, 1)
    # denoised_image3 = median_WMF(noised_image, 1)
    denoised_image4 = filtru_liniar_adaptiv_distanta_MM(noised_image, 1)

    show_image_rgb(noised_image)
    show_image_rgb(denoised_image1)
    # show_image_rgb(denoised_image2)
    # show_image_rgb(denoised_image3)
    show_image_rgb(denoised_image4)


if __name__ == '__main__':
    main()
    plt.show()
