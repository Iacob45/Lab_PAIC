from matplotlib import pyplot as plt
from skimage import io

from Core.constants import IMAGE_PATH
from Core.denoise import median_marginal, median_vectorial
from Core.noise import pune_zgomot_impulsiv
from Core.utilities import show_image_rgb


def main():
    I = io.imread(IMAGE_PATH)
    I = I[:, :, :3]

    show_image_rgb(I)
    noised_image = pune_zgomot_impulsiv(I, 0.3, 0.3)
    denoised_image1 = median_marginal(noised_image, 5)
    denoised_image2 = median_vectorial(noised_image, 5)

    show_image_rgb(noised_image)
    show_image_rgb(denoised_image1)
    show_image_rgb(denoised_image2)



if __name__ == '__main__':
    main()
    plt.show()
