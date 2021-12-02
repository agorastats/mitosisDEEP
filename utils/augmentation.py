from scipy.ndimage.filters import gaussian_filter
from utils.preprocess import Normalizer

import math
import cv2
import numpy as np


REF_IMAGE = 'a'

def to_tuple(param, low=None):
    if isinstance(param, tuple):
        return param
    else:
        return (-param if low is None else low, param)


def rotate(img, angle=30, k_limit=(1, 6), seed=None):
    if seed is None:
        random_state = np.random.RandomState(1234)
    else:
        random_state = np.random.RandomState(seed)
    angle = angle * random_state.randint(k_limit[0], k_limit[1])
    height, width = img.shape[0:2]
    mat = cv2.getRotationMatrix2D((width/2, height/2), angle, 1.0)
    img = cv2.warpAffine(np.float32(img), mat, (width, height),
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return img


def blur(img, k_max=5, seed=None):
    if seed is None:
        random_state = np.random.RandomState(1234)
    else:
        random_state = np.random.RandomState(seed)

    k_limit = to_tuple(k_max, low=1)
    ksize = random_state.randint(k_limit[0], k_limit[1])
    return cv2.blur(img, (ksize, ksize))


def elastic_transform(image, alpha=1, sigma=30, alpha_affine=3, seed=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if seed is None:
        random_state = np.random.RandomState(1234)
    else:
        random_state = np.random.RandomState(seed)

    shape = image.shape
    shape_size = shape[:2]


    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    alpha = float(alpha)
    sigma = float(sigma)
    alpha_affine = float(alpha_affine)

    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)

    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = np.float32(gaussian_filter((random_state.rand(*shape_size) * 2 - 1), sigma) * alpha)
    dy = np.float32(gaussian_filter((random_state.rand(*shape_size) * 2 - 1), sigma) * alpha)

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

    mapx = np.float32(x + dx)
    mapy = np.float32(y + dy)

    return cv2.remap(image, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

def gaussian_noise(image, var_limit=(10,30), seed=None):
    if seed is None:
        random_state = np.random.RandomState(1234)
    else:
        random_state = np.random.RandomState(seed)
    row, col, ch = image.shape
    var = random_state.randint(var_limit[0], var_limit[1])
    mean = var
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    gauss = (gauss - np.min(gauss)).astype(np.uint8)
    return image.astype(np.int32) + gauss


def distorsion(img, distort_limit=(-0.05, 0.05), shift_limit=(-0.05, 0.05), seed=None):
    """"
    ## unconverntional augmnet ################################################################################3
    ## https://stackoverflow.com/questions/6199636/formulas-for-barrel-pincushion-distortion
    ## https://stackoverflow.com/questions/10364201/image-transformation-in-opencv
    ## https://stackoverflow.com/questions/2477774/correcting-fisheye-distortion-programmatically
    ## http://www.coldvision.io/2017/03/02/advanced-lane-finding-using-opencv/
    ## barrel\pincushion distortion
    """
    if seed is None:
        random_state = np.random.RandomState(1234)
    else:
        random_state = np.random.RandomState(seed)

    shift_limit = to_tuple(shift_limit)
    distort_limit = to_tuple(distort_limit)
    shift_limit = to_tuple(shift_limit)
    k = random_state.uniform(distort_limit[0], distort_limit[1])
    dx = random_state.uniform(shift_limit[0], shift_limit[1])
    dy = random_state.uniform(shift_limit[0], shift_limit[1])
    k = k * 0.00001

    height, width = img.shape[:2]
    dx = dx * width
    dy = dy * height
    x, y = np.mgrid[0:width:1, 0:height:1]
    x = x.astype(np.float32) - width / 2 - dx
    y = y.astype(np.float32) - height / 2 - dy
    theta = np.arctan2(y, x)
    d = (x * x + y * y) ** 0.5
    r = d * (1 + k * d * d)
    map_x = r * np.cos(theta) + width / 2 + dx
    map_y = r * np.sin(theta) + height / 2 + dy
    img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return img


def random_brightness(img, alpha=.2, seed=None):
    if seed is None:
        random_state = np.random.RandomState(1234)
    else:
        random_state = np.random.RandomState(seed)
    limit = to_tuple(alpha)
    alpha = 1 + random_state.uniform(limit[0], limit[1])
    return alpha * img


def random_contrast(img, alpha=.2, seed=None):
    if seed is None:
        random_state = np.random.RandomState(1234)
    else:
        random_state = np.random.RandomState(seed)
    limit = to_tuple(alpha)
    alpha = 1 + random_state.uniform(limit[0], limit[1])
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
    return alpha * img + gray


def shift_scale_rotate(img, shift_limit=0.2, scale_limit=0.1, rotate_limit=45, seed=None):
    if seed is None:
        random_state = np.random.RandomState(1234)
    else:
        random_state = np.random.RandomState(seed)

    shift_limit = to_tuple(shift_limit)
    scale_limit = to_tuple(scale_limit)
    rotate_limit = to_tuple(rotate_limit)
    angle = random_state.uniform(rotate_limit[0], rotate_limit[1])
    scale = random_state.uniform(1 + scale_limit[0], 1 + scale_limit[1])
    dx = random_state.uniform(shift_limit[0], shift_limit[1])
    dy = random_state.uniform(shift_limit[0], shift_limit[1])

    height, width = img.shape[:2]

    cc = math.cos(angle/180*math.pi) * scale
    ss = math.sin(angle/180*math.pi) * scale
    rotate_matrix = np.array([[cc, -ss], [ss, cc]])

    box0 = np.array([[0, 0], [width, 0],  [width, height], [0, height], ])
    box1 = box0 - np.array([width/2, height/2])
    box1 = np.dot(box1, rotate_matrix.T) + np.array([width/2+dx*width, height/2+dy*height])

    box0 = box0.astype(np.float32)
    box1 = box1.astype(np.float32)
    mat = cv2.getPerspectiveTransform(box0, box1)
    img = cv2.warpPerspective(img, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return img


def stain_normalizer(ref_image):
    stain_norm = Normalizer()
    stain_norm.fit(cv2.imread(ref_image))
    return stain_norm


def aug_generator(image, mask, seed=None, stain_norm=None):
    if seed is None:
        random_state = np.random.RandomState(1234)
    else:
        random_state = np.random.RandomState(seed)
    # preprocess
    if stain_norm is not None:
        image = stain_norm.transform(image)

    p = random_state.uniform(0, 1.)
    # noise
    if p < 0.5:
        image = gaussian_noise(image, seed=seed)

    # rotation
    image = rotate(image, seed=seed)
    mask = rotate(mask, seed=seed)

    # blur
    if p < 0.3:
        image = blur(image, seed=seed)

    # transformation
    # shift scale
    if p < 0.7:
        image = shift_scale_rotate(image, seed=seed)
        mask = shift_scale_rotate(mask, seed=seed)

    # distorsion
    if p < 0.5:
        f = np.random.choice([elastic_transform, distorsion], 1)[0]
        image = f(image, seed=seed)
        mask = f(mask, seed=seed)

    # contrast or brightness
    if p < 0.4:
        f = np.random.choice([random_brightness, random_contrast], 1)[0]
        image = f(image, seed=seed)

    return image, mask