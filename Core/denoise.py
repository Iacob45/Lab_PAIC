import numpy as np
import scipy


def HE(img_in):
    [L, C] = np.shape(img_in)
    if np.max(img_in) > 1:
        img_in = img_in/255.0

    h, _ = np.histogram(img_in, bins=256, range=(0, 1.0000001))
    h = h/np.sum(h)
    H = np.cumsum(h)

    img_out = np.zeros((L, C))

    for l in range(0, L):
        for c in range(0, C):
            img_out[l, c] = H[np.uint8(255*img_in[l, c])]

    return img_out


def HE_with_window(img_in, w):
    img_in = np.float64(img_in)
    [L, C] = np.shape(img_in)
    if np.max(img_in) > 1:
        img_in = img_in/255.0

    img_out = np.zeros((L - 2*w, C - 2*w))

    for l in range(w, L-w):
        for c in range(w, C-w):
            window = img_in[l-w: l+w+1, c-w: c+w+1].copy()
            new_window = HE(window)
            img_out[l - w, c - w] = new_window[w, w]

    kernel = np.zeros((2*w+1, 2*w+1))
    kernel[w, w] = 1
    img_out = scipy.signal.convolve2d(img_out, kernel, mode='full', boundary='symm')
    img_out = img_out[:L, :C]

    return img_out


def CLHE(img_in, limit):
    [L, C] = np.shape(img_in)
    if np.max(img_in) > 1:
        img_in = img_in/255.0

    h, _ = np.histogram(img_in, bins=256, range=(0, 1.0000001))
    h = h/np.sum(h)

    # limitarea maximelor histogramei + redistribuire in zone scazute
    collect = 0
    for i in range(0, 256):
        if h[i] > limit:
            # tai ce e deasupra, redistribui la sfarsit
            collect += h[i] - limit
            h[i] = limit
    h += collect/256.0

    H = np.cumsum(h)

    img_out = np.zeros((L, C))

    for l in range(0, L):
        for c in range(0, C):
            img_out[l, c] = H[np.uint8(255*img_in[l, c])]

    return img_out, h


def retrieve_window(image, center: tuple[int, int], w_size):
    l = center[0]
    c = center[1]
    return image[l - w_size:l + w_size + 1, c - w_size:c + w_size + 1]


def retrieve_kernel(w_size: int = 3, kernel_type: int = None):
    kernel_size = w_size * 2 + 1

    if kernel_type == 1:
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[w_size, :] = np.float64(1/kernel_size)

    elif kernel_type == 2:
        if w_size != 1:
            raise ValueError("w_size must be 1 for this kernel_type")
        kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])/16.0

    else:
        kernel = np.ones((kernel_size, kernel_size)) / kernel_size ** 2

    return kernel


def apply_conv_filter(image, w_size: int = 3, kernel_type: int = None):
    kernel = retrieve_kernel(w_size, kernel_type)

    [L, C] = np.shape(image)

    image_out = np.zeros(np.shape(image))
    for l in range(0 + w_size, L - w_size):
        for c in range(0 + w_size, C - w_size):
            window = retrieve_window(image, (l, c), w_size)
            image_out[l, c] = np.sum(kernel*window)

    return image_out


def lexicographic_histogram_equalization(image):
    image_shape = image.shape

    image_avg = scipy.signal.convolve2d(image, np.ones((3, 3)) / 9.0, 'same', 'symm')

    image_avg_flat = np.ndarray.flatten(image_avg)
    image_flat = np.ndarray.flatten(image)

    indexes_array = np.lexsort((image_avg_flat, image_flat))
    pixels_per_level = len(indexes_array) // 256
    print(pixels_per_level)

    image_out = np.zeros(np.shape(image_flat))

    for i, index in enumerate(indexes_array):
        gray_level = min(i/pixels_per_level, 255)
        image_out[index] = gray_level/255.0




    image_out = np.reshape(image_out, image_shape)

    return image_out


def median_marginal(img, W):
    if np.max(img) > 1:
        img = img / 255.0

    [X, Y, Z] = np.shape(img)
    dim = 2 * W + 1
    kernel = np.zeros((dim, dim))
    kernel[W, W] = 1

    img_out = np.zeros((X, Y, Z))
    for z in range(Z):
        plane = img[:, :, z]
        img_bordat = scipy.signal.convolve2d(plane, kernel, "full", "symm")

        for x in range(W, X + W):
            for y in range(W, Y + W):
                valori_selectate = retrieve_window(img_bordat, (x, y), W)
                valori_ordonate = np.sort(valori_selectate, axis=None)
                median = valori_ordonate[(dim * dim - 1) // 2]
                img_out[x - W, y - W, z] = median

    return img_out


def median_vectorial(img, W):
    if np.max(img) > 1:
        img = img / 255.0

    [X, Y, Z] = np.shape(img)
    dim = 2 * W + 1
    kernel = np.zeros((dim, dim))
    kernel[W, W] = 1

    img_bordat = np.zeros((X + dim - 1, Y + dim - 1, Z))
    img_bordat[:, :, 0] = scipy.signal.convolve2d(img[:, :, 0], kernel, "full", "symm")
    img_bordat[:, :, 1] = scipy.signal.convolve2d(img[:, :, 1], kernel, "full", "symm")
    img_bordat[:, :, 2] = scipy.signal.convolve2d(img[:, :, 2], kernel, "full", "symm")

    img_out = np.zeros((X, Y, Z))
    for x in range(W, X + W):
        for y in range(W, Y + W):
            valori_selectate = np.zeros((dim, dim, Z))
            valori_selectate[:, :, 0] = retrieve_window(img_bordat[:, :, 0], (x, y), W)
            valori_selectate[:, :, 1] = retrieve_window(img_bordat[:, :, 1], (x, y), W)
            valori_selectate[:, :, 2] = retrieve_window(img_bordat[:, :, 2], (x, y), W)

            median_r = np.sort(valori_selectate[:, :, 0], axis=None)[(dim * dim - 1) // 2]
            median_g = np.sort(valori_selectate[:, :, 1], axis=None)[(dim * dim - 1) // 2]
            median_b = np.sort(valori_selectate[:, :, 2], axis=None)[(dim * dim - 1) // 2]

            median_marginal = np.array((median_r, median_g, median_b))
            pixels = valori_selectate.reshape(-1, 3)
            distances = np.linalg.norm(pixels - median_marginal, axis=1)
            closest_pixel = pixels[np.argmin(distances)]

            median_vectorial = tuple(closest_pixel)

            img_out[x - W, y - W, :] = median_vectorial

    return img_out
