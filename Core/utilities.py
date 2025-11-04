import numpy as np
import scipy
from matplotlib import pyplot as plt


def show_image_rgb(image, vmax: int = None, autoscale: bool = False):
    plt.figure()
    if autoscale:
        plt.imshow(image)
    elif vmax:
        plt.imshow(image, vmin=0, vmax=vmax)
    elif np.max(image) > 1:
        plt.imshow(image, vmin=0, vmax=255)
    else:
        plt.imshow(image, vmin=0, vmax=1)


    plt.colorbar()


def show_image_gray(image, vmax: int = None, autoscale: bool = False):
    plt.figure()
    if autoscale:
        plt.imshow(image, cmap='gray')
    elif vmax:
        plt.imshow(image, cmap='gray', vmin=0, vmax=vmax)
    elif np.max(image) > 1:
        plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    else:
        plt.imshow(image, cmap='gray', vmin=0, vmax=1)

    plt.colorbar()


def show_plot(*args, title: str = "Plot", color: str | int = 'black', reuse_figure: bool = False):
    if not reuse_figure:
        plt.figure(figsize=(8, 4))
    else:
        color = np.random.rand(3, )

    if len(args) == 2:
        plt.plot(args[0], args[1], color=color)
    elif len(args) == 1:
        plt.plot(args[0], color=color)
    else:
        raise ValueError

    plt.title(title)
    plt.xlabel('Nivel de gri')
    plt.ylabel('Număr pixeli')
    plt.grid(True)


def show_hist(image, bins: int = 100, normalize: bool = True, title: str = 'Histograma'):
    hist, bins = np.histogram(image, bins=bins)
    if normalize:
        hist = hist / np.sum(hist)
    show_plot(bins[:len(bins)-1], hist, title=title)


def T(u, r):
    if r < 0:
        raise ValueError

    v = u**r

    return v


def expandare_liniara(u, a, b):
    if u < a:
        v = 0
    elif u > b:
        v = 1
    else:
        v = (u - a) / (b - a)

    return v


def prelucrare_scalara(img_in):
    if np.max(img_in) > 1:
        img_in = img_in/255.0

    [L, C] = np.shape(img_in)
    img_out = np.zeros((L, C))

    a = np.percentile(img_in, 5)
    b = np.percentile(img_in, 95)

    for l in range(0, L):
        for c in range(0, C):
            img_out[l, c] = expandare_liniara(img_in[l, c], a, b)

    return img_out


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


def MAE(imgref, img):
    if imgref.shape != img.shape:
        raise ValueError("Different image shapes.")
    return np.average(np.abs(img - imgref))


def NMAE(imgref, img):
    if imgref.shape != img.shape:
        raise ValueError("Different image shapes.")
    return np.average(np.abs(img - imgref)/imgref)


def MSE(imgref, img):
    if imgref.shape != img.shape:
        raise ValueError("Different image shapes.")
    return np.average((img - imgref)**2)
