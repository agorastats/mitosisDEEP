import json

import cv2
import numpy as np
import pandas as pd

from utils.image import show_image, read_image
from utils.loadAndSaveResults import read_data_frame, read_json


def convert_coco_json_to_csv(filename):
    import pandas as pd
    import json

    # COCO2017/annotations/instances_val2017.json
    s = json.load(open(filename, 'r'))
    out_file = filename[:-5] + '.csv'
    out = open(out_file, 'w')
    out.write('id,x1,y1,x2,y2,label\n')

    all_ids = []
    for im in s['images']:
        all_ids.append(im['id'])

    all_ids_ann = []
    for ann in s['annotations']:
        image_id = ann['image_id']
        all_ids_ann.append(image_id)
        x1 = ann['bbox'][0]
        x2 = ann['bbox'][0] + ann['bbox'][2]
        y1 = ann['bbox'][1]
        y2 = ann['bbox'][1] + ann['bbox'][3]
        label = ann['category_id']
        out.write('{},{},{},{},{},{}\n'.format(image_id, x1, y1, x2, y2, label))

    all_ids = set(all_ids)
    all_ids_ann = set(all_ids_ann)
    no_annotations = list(all_ids - all_ids_ann)
    # Output images without any annotations
    for image_id in no_annotations:
        out.write('{},{},{},{},{},{}\n'.format(image_id, -1, -1, -1, -1, -1))
    out.close()

    # Sort file by image id
    s1 = pd.read_csv(out_file)
    s1.sort_values('id', inplace=True)
    s1.to_csv(out_file, index=False)

# download data: https://zenodo.org/record/4643381#.Yiy7tBuCE5n
def get_bbox_df(image_folder="/drive/MyDrive/MIDOG_Challenge/images",
                annotation_file="ideas/MIDOG.json"):
    hamamatsu_rx_ids = list(range(0, 51))
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

    df = pd.DataFrame(rows, columns=["file_name", "image_id", "width", "height", "box", "point", "cat", "scanner"])
    return (df)



if __name__ == '__main__':

    bbox_df = get_bbox_df(annotation_file="ideas/MIDOG.json")
    # convert_coco_json_to_csv('ideas/MIDOG.json')
    # https://stackoverflow.com/questions/65138694/opencv-blob-defect-anomaly-detection
    img = cv2.imread("sample_data/03x.jpg")
    show_image(img)
    mean = np.mean(img, axis=(0, 1))
    std = np.std(img, axis=(0, 1))
    mask = (np.abs(img - mean) / std >= 4.5).any(axis=2)
    mask_u8 = mask.astype(np.uint8) * 255
    show_image(mask_u8)
    # todo: si pasem a thresh binari funcionara millor !
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    candidates = sorted(contours, key=cv2.contourArea, reverse=True)
    # cv2.drawContours(img, [each candidates coords], -1, (255, 0, 0), 3)
    pass

    patchesInfoDF = read_data_frame('patches/20220207/infoDF.csv')

    resultDict = dict()
    for i, item in patchesInfoDF.iterrows():
        img = read_image('patches/20220207/masks/%s' % item['id'])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        resultDict[item['id']] = len(contours)
    patchesInfoDF.loc[:, 'mitosisInPatch'] = patchesInfoDF.loc[:, 'id'].map(resultDict)
    pass
