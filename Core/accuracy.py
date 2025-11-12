import numpy as np


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
