import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from os import makedirs, path
from PIL import Image

MAX_SIZE = 255.


def image_normalize(img):
    img = np.asarray(img, dtype=np.float64)
    if img.max() > 1.0:
        img /= MAX_SIZE
    return img


def img2np(img):
    img = np.asarray(img)
    if len(img.shape) is not 3:
        raise Exception("img2np function works with RGB images.")
    # Channel, Breadth, Height or Channel, Width, Height
    return img.transpose(2, 1, 0)


def read_image(filename, size):
    assert os.path.exists(filename), 'Problems reading filename %s' % filename
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # need to change to train in RGB format the nets
    image = cv2.resize(image, (size, size))
    return image


def read_mask(filename, size):
    assert os.path.exists(filename), 'Problems reading filename %s' % filename
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (size, size))
    (thresh, black_white_image) = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return black_white_image


def read_and_normalize_image(filename, convert=None):
    assert os.path.exists(filename), 'Problems reading filename %s' % filename
    image = cv2.imread(filename)
    if convert is not None:
        image = cv2.cvtColor(image, convert)
    image = image_normalize(image)
    return image


def np2img(np_array):
    np_array = np.asarray(np_array)
    if len(np_array.shape) is not 3:
        raise Exception("np2img function works with 3 dimensional numpy arrays than can be converted to RGB image.")
    return np_array.transpose(2, 1, 0)


def print_files_with_extension(folder, extension, recursive=True):
    for filename in glob.iglob(os.path.abspath(folder) + '/' + '**/*.' + extension, recursive=recursive):
        print(filename)


def reduce_size_of_image_and_save_it(filename, outputDir):
    img = Image.open(filename).convert('RGB')
    img.save(outputDir + filename, format='jpeg', quality=10, optimize=True)


def create_dir(output_dir):
    if not path.exists(output_dir):
        makedirs(output_dir)


def create_mask_with_annotations_polynomial(image, annotations_list):
    mask_image = np.zeros_like(image)
    for m in range(len(annotations_list)):
        mask_image = cv2.fillPoly(mask_image, pts=[np.array(annotations_list[m])], color=(255, 255, 255))
        mask_image = cv2.GaussianBlur(mask_image, ksize=(1, 1), sigmaX=0, sigmaY=0)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    return mask_image


def create_mask_with_annotations_circle(image, annotations_list, radius=15):
    mask_image = np.zeros_like(image)
    for m in range(len(annotations_list)):
        mask_image = cv2.circle(mask_image, tuple(annotations_list[0]), radius, color=(255, 255, 255), thickness=-1)
        mask_image = cv2.GaussianBlur(mask_image, ksize=(1, 1), sigmaX=0, sigmaY=0)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    return mask_image


def generate_patch(image, height, width, centered, patch_size=256):
    patch_center = np.array([height, width])
    limit_image = image.shape
    # center patch if possible, else contains patch that starts at border of image
    patch_x = int(patch_center[0] - patch_size / centered[0]) if patch_center[0] > patch_size / 2. else 0
    patch_y = int(patch_center[1] - patch_size / centered[1]) if patch_center[1] > patch_size / 2. else 0
    # to ensure patch size
    patch_x -= max(0, patch_x + patch_size - limit_image[0])
    patch_y -= max(0, patch_y + patch_size - limit_image[1])
    patch_image = image[patch_x:patch_x + patch_size, patch_y:patch_y + patch_size]
    return patch_image


def show_image(*args):
    if len(args) == 1:
        plt.imshow(args[0])
    else:
        f, ax = plt.subplots(1, len(args), figsize=(8, 8))
        for i, arg in enumerate(args):
            ax[i].imshow(arg)
    plt.show()


def show_images_of_generator_item(images, masks):
    grid_width = 4
    grid_height = int(len(images) / grid_width)
    fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width, grid_height))

    for i, (im, mask) in enumerate(zip(images, masks)):
        ax = axs[int(i / grid_width), i % grid_width]
        ax.imshow(im, cmap='bone')
        ax.imshow(mask.squeeze(), alpha=0.5, cmap="Reds")
        ax.axis('off')
    plt.suptitle("Scanner images | Red: Mitosis mask")
    plt.show()
