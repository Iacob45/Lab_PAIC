import numpy as np
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
