import numpy as np
import matplotlib.pyplot as plt


def show(*img):
    if len(img) < 2:
        show_one(img[0])
        return

    fig, axes = plt.subplots(1, len(img))

    for img_i, i in zip(img, range(len(img))):
        axes[i].imshow(img_i, cmap='gray')
        axes[i].set_title('')

    fig.set_figwidth(20)  # ширина и
    fig.set_figheight(10)  # высота \"Figure\

    #     plt.gray()
    plt.show()


def show_one(img, n=10):
    fig, ax = plt.subplots()
    fig.set_figwidth(n)  # ширина и
    fig.set_figheight(n)  # высота \"Figure\
    plt.imshow(img, cmap='gray')
    plt.show()