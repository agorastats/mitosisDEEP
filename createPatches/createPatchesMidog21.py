import logging
import os
import cv2
import numpy as np
import pandas as pd

from utils.loadAndSaveResults import read_json
from utils.runnable import Main
from createPatches.base import CreatePatches
from utils.image import create_mask_with_annotations_circle, create_shape_mask_inferring_from_centroid_annotations

# data images (training data set) downloaded from: https://zenodo.org/record/4643381#.Yiy7tBuCE5n
DATA_PATH = 'data/MIDOG'  # path that contains inside it training images


def get_midog_bbox_info_df(annotation_file="ideas/MIDOG.json", only_mitotic=True):
    # hamamatsu_rx_ids = list(range(0, 51))
    hamamatsu_360_ids = list(range(51, 101))
    aperio_ids = list(range(101, 151))
    leica_ids = list(range(151, 201))

    rows = []

    data = read_json(annotation_file)
    categories = {1: 'mitotic figure', 2: 'not mitotic figure'}

    for row in data["images"]:
        file_name = row["file_name"]
        image_id = row["id"]
        width = row["width"]
        height = row["height"]

        scanner = "Hamamatsu XR"
        if image_id in hamamatsu_360_ids:
            scanner = "Hamamatsu S360"
        if image_id in aperio_ids:
            scanner = "Aperio CS"
        if image_id in leica_ids:
            scanner = "Leica GT450"

        for annotation in [anno for anno in data['annotations'] if anno["image_id"] == image_id]:
            box = annotation["bbox"]
            cat = categories[annotation["category_id"]]
            point = [0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])]
            rows.append([file_name, image_id, width, height, box, point, cat, scanner])

    info_df = pd.DataFrame(rows,
                           columns=["file_name", "image_id", "width", "height", "box", "point", "cat", "scanner"])
    if only_mitotic:
        info_df = info_df.loc[info_df.loc[:, 'cat'] == 'mitotic figure', :]
    return info_df


class CreatePatchesMidog21(CreatePatches):
    def __init__(self):
        super().__init__()
        self.prefix_img = 'midog21'
        self.data_path = DATA_PATH
        self.annot_path = self.data_path
        self.annot_file = 'MIDOG.json'
        self.img_format = '.tiff'
        self.patchify = True

    def get_annotations(self, annot_df, name_img):
        result = []
        try:
            aux_df = annot_df.loc[annot_df.loc[:, 'file_name'] == name_img + self.img_format, :]
            for i, annot in aux_df.iterrows():
                result.append(list(map(int, annot['point'])))
        except:
            raise Warning("name_img %s has invalid value in annot_df" % name_img)
        return result

    def run(self, options):
        images_list = [f for f in os.listdir(self.data_path) if f.endswith(self.img_format)]
        annot_df = get_midog_bbox_info_df(os.path.join(self.annot_path, self.annot_file))
        for j, img in enumerate(sorted(images_list)):
            if j % 100 == 0:
                logging.info('(progress) Iterating image number: %i / %i' % (j, len(images_list)))
            logging.info('___prepare patches for img: %s' % str(img))
            name_img = img.split('.')[0]
            image = cv2.imread(os.path.join(self.data_path, img))
            annot_list = self.get_annotations(annot_df, name_img)
            # mask = create_mask_with_annotations_circle(image, annot_list)
            mask = create_shape_mask_inferring_from_centroid_annotations(image, annot_list)
            assert len(np.unique(mask)) <= 2, 'more than 2 color pixels'

            self.create_patches_with_annotations(image, mask, annot_list, name_img, patch_size=options['patch_size'])

            self.create_patches_with_patchify(image, mask, name_img, patch_size=options['patch_size'])


if __name__ == '__main__':
    Main(
        CreatePatchesMidog21()
    ).run()
