from functools import cache

import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt


def test_matrice():
    # 1. Creare matrici
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    B = np.array([[9, 8, 7],
                  [6, 5, 4],
                  [3, 2, 1]])

    print("Matrice A:\n", A)
    print("Matrice B:\n", B)

    # 2. Adunare și scădere
    print("\nA + B:\n", A + B)
    print("\nA - B:\n", A - B)

    # 3. Înmulțire element cu element și înmulțire de matrici
    print("\nA * B (element-wise):\n", A * B)
    print("\nA @ B (matmul):\n", A @ B)

    # 4. Transpusa și inversa (inversa doar pe submatrice 2x2)
    print("\nTranspusa lui A:\n", A.T)
    C = np.array([[2, 1], [5, 3]])
    print("\nMatrice C:\n", C)
    print("Inversa lui C:\n", np.linalg.inv(C))

    # 5. Determinant și rang
    print("\nDeterminantul lui C:", np.linalg.det(C))
    print("Rangul lui A:", np.linalg.matrix_rank(A))

    # 6. Normă și valori proprii
    print("\nNorma lui A:", np.linalg.norm(A))
    eigvals, eigvecs = np.linalg.eig(C)
    print("Valori proprii ale lui C:", eigvals)
    print("Vectori proprii ai lui C:\n", eigvecs)

    # 7. Indexare și slicing
    print("\nElementul [1,2] din A:", A[1, 2])  # linia 1, coloana 2
    print("Linia 2 din B:", B[1, :])
    print("Coloana 3 din A:", A[:, 2])

    # 8. Reshape și concatenare
    print("\nA reshape (1x9):\n", A.reshape(1, 9))
    print("Concat [A | B]:\n", np.hstack((A, B)))
    print("Concat [A peste B]:\n", np.vstack((A, B)))

    # 9. Generare de matrici speciale
    Z = np.zeros((3, 3))
    I = np.eye(3)
    R = np.random.randint(1, 10, (3, 3))
    print("\nMatrice zero:\n", Z)
    print("Matrice identitate:\n", I)
    print("Matrice random:\n", R)


def lena():
    image = io.imread('lena.png')

    # print(np.shape(image))
    # print(type(image))
    # print(np.min(image[:, :, 2]))

    culori = ((0, "red"), (1, "green"), (0, "blue"))

    for culoare in culori:
        print(culoare[1], np.min(image[:, :, culoare[0]]), np.max(image[:, :, culoare[0]]))


def plots():
    image = io.imread('lena.png')

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 rând, 3 coloane

    channels = ["Red channel", "Green channel", "Blue channel"]

    for i, ax in enumerate(axes):
        im = ax.imshow(image[:, :, i], cmap='Reds_r', vmin=0, vmax=255)  # fără cmap -> "viridis" default
        ax.set_title(channels[i])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 rând, 3 coloane

    channels = ["Red", "Green", "Blue"]
    cmaps = ["Reds", "Greens", "Blues"]

    for i, ax in enumerate(axes):
        im = ax.imshow(image[:, :, i], cmap=cmaps[i])
        ax.set_title(channels[i])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


def plots_float():
    image = io.imread('lena.png')
    r = image[:, :, 0].astype(float) / 255.0

    plt.imshow(r, cmap='Reds_r', vmin=0, vmax=1)
    plt.colorbar()
    plt.title("Sarmale")
    plt.show()


def plots_color():
    image = io.imread('lena.png')

    gray = color.rgb2gray(image)
    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]
    yy = 0.3 * red + 0.3 * green + 0.4 * blue
    yyy = yy / 255.0
    y_sth = yyy / 4.0 + 0.2

    [linii, coloane] = np.shape(yyy)
    out = np.zeros((linii, coloane))

    for i in range(linii):
        for j in range(coloane):
            out[i, j] = LUT[np.uint8(255.0 * yyy[i, j])].item() / 255.0

    plt.imshow(out, cmap='gray', vmin=0, vmax=1)
    plt.colorbar()
    plt.title("Sarmale")
    plt.show()


@cache
def functie_simpla(x: int | float) -> float:
    return x / 4.0 + 0.2


LUT = np.zeros((256, 1))
for c in range(0, 256):
    LUT[c] = functie_simpla(c)


if __name__ == "__main__":
    # test_matrice()
    # lena()
    # plots()
    # plots_float()
    plots_color()
