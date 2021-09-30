import glob
import os
from os import makedirs, path
import re

import cv2
import numpy as np
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


def read_image(filename):
    assert os.path.exists(filename), 'Problems reading filename %s' % filename
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image_normalize(image)
    return img2np(image)


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


def load_annotations(path):
    assert re.compile(r'.*\.csv').match(path) is not None
    result = []
    ln = 0
    for line in open(path).readlines():
        ln += 1
        # Ignore empty lines
        if len(line.strip()) == 0:
            continue
        # Parse line into list of numbers
        points = list(map(lambda x: x, line.strip().split(',')))
        try:
            result.append([[int(points[i]), int(points[i + 1])] for i in range(0, len(points), 2)])
        except:
            raise Warning("Line %d in %s has invalid value." % (ln, path))
    return result


def create_output_dir(output_dir):
    if not path.exists(output_dir):
        makedirs(output_dir)


def create_mask_with_annotations(image, annotations_list):
    mask_image = np.zeros_like(image)
    for m in range(len(annotations_list)):
        mask_image = cv2.fillPoly(mask_image, pts=[np.array(annotations_list[m])], color=(225, 255, 255))
        mask_image = cv2.GaussianBlur(mask_image, ksize=(0, 0), sigmaX=2, sigmaY=2)
    return mask_image