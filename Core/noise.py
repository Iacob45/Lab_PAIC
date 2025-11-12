import numpy as np


def pune_zgomot_impulsiv(image, p0, p1):
    if np.max(image) > 1:
        image = image / 255.0

    [x, y, z] = np.shape(image)
    coord_x_0 = np.random.randint(0, x, int(p0 * x * y * z))
    coord_y_0 = np.random.randint(0, y, int(p0 * x * y * z))
    coord_z_0 = np.random.randint(0, z, int(p0 * x * y * z))
    image[coord_x_0, coord_y_0, coord_z_0] = 0

    coord_x_1 = np.random.randint(0, x, int(p1 * x * y * z))
    coord_y_1 = np.random.randint(0, y, int(p1 * x * y * z))
    coord_z_1 = np.random.randint(0, z, int(p1 * x * y * z))
    image[coord_x_1, coord_y_1, coord_z_1] = 1

    return image
