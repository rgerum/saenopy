import numpy as np
from PIL import Image
from scipy.ndimage import shift
from skimage.registration import phase_cross_correlation


def normalizing(img):
    img = img - np.percentile(img, 1)  # 1 Percentile
    img = img / np.percentile(img, 99.99)  # norm to 99 Percentile
    img[img < 0] = 0.0
    img[img > 1] = 1.0
    return img


def cropping_after_shift(image, shift_x, shift_y):
    if shift_x <= 0:
        image = image[:, int(np.ceil(-shift_x)) :]
    else:
        image = image[:, : -int(np.ceil(shift_x))]
    if shift_y <= 0:
        image = image[int(np.ceil(-shift_y)) :, :]
    else:
        image = image[: -int(np.ceil(shift_y)), :]
    return np.array(image, dtype=float)


def correct_stage_drift(image1, image2, additional_images=None):
    """
    # correcting frame shift between images of beads before and after cell removal.

    # the correction is done by finding the shift between two images using image registration. Then the images are
    # cropped to the common field of view. If this script finds further images of the cells, it wil also crop them to
    # this field of view. The output is saved to the input folder. For each "experiment" a new folder is created. An
    # experiment is identified as a directory that contains one folder for the images before cell removal and one folder
    # with images after the cell removal.

    :param image1:
    :param image2:
    :param additional_images:
    :return:
    """
    if additional_images is None:
        additional_images = []

    # find shift with image registration
    shift_values = phase_cross_correlation(image1, image2, upsample_factor=100)

    shift_y = shift_values[0][0]
    shift_x = shift_values[0][1]

    # using interpolation to shift subpixel precision, image2 is the reference
    image1_shift = shift(image1, shift=(-shift_y, -shift_x), order=5)

    # normalizing and converting to image format
    b = normalizing(cropping_after_shift(image1_shift, shift_x, shift_y))
    a = normalizing(cropping_after_shift(image2, shift_x, shift_y))
    b_save = Image.fromarray(b * 255)
    a_save = Image.fromarray(a * 255)

    # doing the same with additional images
    additional_images_save = []
    for add_image in additional_images:
        add_image_shift = shift(add_image, shift=(-shift_y, -shift_x), order=5)
        add_image_norm = normalizing(
            cropping_after_shift(add_image_shift, shift_x, shift_y)
        )
        add_image_save = Image.fromarray(add_image_norm * 255)
        additional_images_save.append(add_image_save)

    return b_save, a_save, additional_images_save, (shift_x, shift_y)
