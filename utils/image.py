import glob
import io
import logging
import os
import cv2
import base64
import numpy as np
import matplotlib.pyplot as plt

from os import makedirs, path
from PIL import Image
from skimage.measure import label, regionprops

MAX_SIZE = 255.


def image_normalize(img):
    img = np.asarray(img, dtype=np.float64)
    if img.max() > 1.0:
        img /= MAX_SIZE
    return img


def img2np(img):
    img = np.asarray(img)
    if len(img.shape) != 3:
        raise Exception("img2np function works with RGB images.")
    # Channel, Breadth, Height or Channel, Width, Height
    return img.transpose(2, 1, 0)


def read_image(filename, size=None):
    assert os.path.exists(filename), 'Problems reading filename %s' % filename
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # need to change to train in RGB format the nets
    if size:
        image = cv2.resize(image, (size, size))
    return image


def read_mask(filename, size=None):
    assert os.path.exists(filename), 'Problems reading filename %s' % filename
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if size:
        image = cv2.resize(image, (size, size))
    (thresh, black_white_image) = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)   # binarize it
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
    if len(np_array.shape) != 3:
        raise Exception("np2img function works with 3 dimensional numpy arrays than can be converted to RGB image.")
    return np_array.transpose(2, 1, 0)


def print_files_with_extension(folder, extension, recursive=True):
    for filename in glob.iglob(os.path.abspath(folder) + '/' + '**/*.' + extension, recursive=recursive):
        print(filename)


def reduce_size_of_image_and_save_it(filename, outputDir):
    img = Image.open(filename).convert('RGB')
    img.save(outputDir + filename, format='jpeg', quality=10, optimize=True)


def create_dir(dir_path):
    '''
        Create directory if not exists
    '''
    if not path.exists(dir_path):
        logging.info('__create new folder: %s' % dir_path)
        makedirs(dir_path)


def create_mask_with_annotations_polynomial(image, annotations_list):
    '''
    Generate mask with multiple annotations
    '''
    mask_image = np.zeros_like(image)
    for m in range(len(annotations_list)):
        mask_image = cv2.fillPoly(mask_image, pts=[np.array(annotations_list[m])], color=(255, 255, 255))
        mask_image = cv2.GaussianBlur(mask_image, ksize=(1, 1), sigmaX=0, sigmaY=0)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    return mask_image


def create_mask_with_annotations_circle(image, annotations_list, radius=15):
    '''
    Generate mask with annotations as circles (1 annotation, for ex: centroid of interest object
    '''
    mask_image = np.zeros_like(image)
    for m in range(len(annotations_list)):
        mask_image = cv2.circle(mask_image, tuple(annotations_list[m]), radius, color=(255, 255, 255), thickness=-1)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    return mask_image


def generate_patch(image, height, width, centered, patch_size=256):
    '''
    Generate patches given some image
    '''
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
    '''
    Plot all images elements of input
    '''
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


def rle_encode(img):
    '''
    # ref: https://www.kaggle.com/stainsby/fast-tested-rle
    # ref:  https://www.kaggle.com/paulorzp/run-length-encode-and-decode
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


def maskInColor(image: np.ndarray,
                mask: np.ndarray,
                color: tuple = (0, 0, 255),
                alpha: float = 0.2):
    image = np.array(image)
    H, W, C = image.shape
    mask = mask.reshape(H, W, 1)
    overlay = image.astype(np.float32)
    overlay = 255 - (255 - overlay) * (1 - mask * alpha * color / 255)
    overlay = np.clip(overlay, 0, 255)
    overlay = overlay.astype(np.uint8)
    return overlay


def trace_boundingBox(image: np.ndarray,
                      mask: np.ndarray,
                      color: tuple = (255, 0, 0),  # red on rgb
                      width: int = 10,
                      expand_box_px: int = 30):
    """
    Draw a bounding box on image

     Parameter
     ----------
     image : image on which we want to draw the box
     mask  : mask to process
     color : color we want to use to draw the box edges
     width : box edges's width
     expand_box_px: expand some pixels on bounding box

    """

    lbl = label(mask)
    props = regionprops(lbl)
    for prop in props:
        #** bbox **: tuple, Bounding box ``(min_row, min_col, max_row, max_col)``
        coin1 = (min(image.shape[1], prop.bbox[3] + expand_box_px), min(image.shape[0], prop.bbox[2] + expand_box_px))
        coin2 = (max(0, prop.bbox[1] - expand_box_px), max(0, prop.bbox[0] - expand_box_px))
        # coin1 = (prop.bbox[3], prop.bbox[2])
        # coin2 = (prop.bbox[1], prop.bbox[0])
        cv2.rectangle(image, coin2, coin1, color, width)
    return image


def array_to_b64(img, ext="jpeg"):
    buffer = io.BytesIO()
    Image.fromarray(img).save(buffer, format=ext)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{ext};base64, {encoded}"
