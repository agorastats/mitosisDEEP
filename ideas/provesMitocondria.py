import cv2
from patchify import patchify
import tifffile as tiff
from utils.image import create_dir

# link interessant: https://github.com/bnsreenu/python_for_microscopists/blob/643521e2ff152ed52369bd40391e226b8a71d481/222_working_with_large_data_that_does_not_fit_memory_semantic_segm/222_unet_loading_data_from_drive.py
# unpatchify: https://github.com/bnsreenu/python_for_microscopists/blob/2c2b120fec17d8686572719916920bc05e3288f8/206_sem_segm_large_images_using_unet_with_patchify.py
# how to train unet on your dataset: https://medium.com/coinmonks/learn-how-to-train-u-net-on-your-dataset-8e3f89fbd623
# unet using keras: https://github.com/zhixuhao/unet

if __name__ == '__main__':
    # ref: https://www.youtube.com/watch?v=7IL7LKSLb9I
    # ref github: https://github.com/bnsreenu/python_for_microscopists/blob/2c2b120fec17d8686572719916920bc05e3288f8/Tips_Tricks_5_extracting_patches_from_large_images_and_masks_for_semantic_segm.py
    output_dir = '../sample_data/proves/prova2'
    create_dir(output_dir + '/images/')
    create_dir(output_dir + '/masks/')

    large_image_stack = tiff.imread('sample_data/mitocondria/image.tif')
    large_mask_stack = tiff.imread('sample_data/mitocondria/mask.tif')

    for img in range(large_image_stack.shape[0]):

        large_image = large_image_stack[img]
        patches_img = patchify(large_image, (256, 256), step=256)  # Step=256 for 256 patches means no overlap

        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):
                single_patch_img = patches_img[i, j, :, :]
                tiff.imwrite(output_dir + '/images/' + str(img) + '_' + str(i) + str(j) + ".tif", single_patch_img)


    for msk in range(large_mask_stack.shape[0]):

        large_mask = large_mask_stack[msk]

        patches_mask = patchify(large_mask, (256, 256), step=256)  # Step=256 for 256 patches means no overlap

        for i in range(patches_mask.shape[0]):
            for j in range(patches_mask.shape[1]):
                single_patch_mask = patches_mask[i, j, :, :]
                tiff.imwrite(output_dir + '/masks/' + str(msk) + '_' + str(i) + str(j) + ".tif", single_patch_mask)
                single_patch_mask = single_patch_mask / 255.

