import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
import imageio
import matplotlib.pylab as pylab

# Set the default figure size for better visualization
pylab.rcParams['figure.figsize'] = (20.0, 7.0)

#   Load an image file and convert it to float data type.
def load_image(filename):
    return imageio.imread(filename).astype(float)

#   Perform 2D Discrete Cosine Transform (DCT) on an array.
def dct2(a):
    return fftpack.dct(fftpack.dct(a, axis=0, norm='ortho'), axis=1, norm='ortho')

#   Perform 2D Inverse Discrete Cosine Transform (IDCT) on an array.
def idct2(a):
    return fftpack.idct(fftpack.idct(a, axis=0, norm='ortho'), axis=1, norm='ortho')

#   Perform 8x8 block DCT on an image.
def perform_dct_on_image(image):
    imsize = image.shape
    dct = np.zeros(imsize)

    for i in np.r_[:imsize[0]:8]:
        for j in np.r_[:imsize[1]:8]:
            dct[i:(i+8), j:(j+8)] = dct2(image[i:(i+8), j:(j+8)])

    return dct

#   Apply threshold to DCT coefficients.
def apply_threshold(dct, thresh):
    return dct * (abs(dct) > (thresh * np.max(dct)))

#   Perform 8x8 block IDCT on the thresholded DCT coefficients.
def perform_idct_on_image(dct):
    imsize = dct.shape
    im_dct = np.zeros(imsize)

    for i in np.r_[:imsize[0]:8]:
        for j in np.r_[:imsize[1]:8]:
            im_dct[i:(i+8), j:(j+8)] = idct2(dct[i:(i+8), j:(j+8)])

    return im_dct

def display_images(images, titles):
    """
    Display a list of images with corresponding titles.
    """
    for i, (image, title) in enumerate(zip(images, titles)):
        plt.figure()
        plt.imshow(image, cmap='gray')
        plt.title(title)

    plt.show()

def display_images_side_by_side(images, titles):
    """
    Display a list of images with corresponding titles.
    """
    fig, axes = plt.subplots(1, len(images), figsize=(20, 7))

    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    plt.show()


def main():
    # Load image
    im = load_image("f14.tif")

    # Perform DCT on the image
    dct = perform_dct_on_image(im)
    display_images([dct], ["DCT Image"])
    # Apply threshold to the DCT coefficients
    thresh = 0.012
    dct_thresh = apply_threshold(dct, thresh)

    # Perform IDCT on the thresholded DCT coefficients
    im_dct = perform_idct_on_image(dct_thresh)

    # Display the original and reconstructed images
    # display_images([im, im_dct], ["Original Image", "DCT Compressed Image"])
    display_images_side_by_side([im, im_dct], ["Original Image", "DCT Compressed Image"])
    
if __name__ == "__main__":
    main()
