import glob
import sys

import nibabel as nib
import skimage
from tqdm import tqdm

sys.path.append('/users/arivajoo/GPAI')
import os


def cut_kidneys(ct, seg):
    kidney_mask = seg.copy()

    kidney_mask[kidney_mask > 0] = 1
    for i in range(10):
        kidney_mask = skimage.morphology.binary_dilation(kidney_mask)
    labeled_image, count = skimage.measure.label(kidney_mask, return_num=True)
    objects = skimage.measure.regionprops(labeled_image)
    if len(objects) > 2:
        print("Too many kidneys!")

    out = []
    for obj in objects:
        z_min, x_min, y_min, z_max, x_max, y_max = obj["bbox"]

        coords_with_margin = slice(z_min - 10, z_max + 10), slice(x_min - 15, x_max + 15), slice(y_min - 15, y_max + 15)

        kidney = ct[coords_with_margin]
        local_seg = seg[coords_with_margin]

        if 2 in local_seg:
            label = "TUMOR"
        else:
            label = "HEALTHY"

        out.append((kidney, local_seg, label))
    return out


def cropping_pipeline(segmentation_paths):
    for seg_path in tqdm(segmentation_paths):
        ct_path = seg_path.replace(".nii.gz", "_0000.nii.gz").replace("labelsTr", "imagesTr")
        if "kidney/tuh" in ct_path:
            ct_path = ct_path.replace("labels", "images")
        case_id = seg_path.split("/")[-1].strip(".nii.gz")

        seg = nib.load(seg_path)
        ct = nib.load(ct_path)

        out = cut_kidneys(ct.get_fdata(), seg.get_fdata())

        dir_name = "kidney_crops"

        seg_path_dir = os.path.dirname(seg_path)
        ct_path_dir = os.path.dirname(ct_path)

        if os.path.exists(os.path.join(seg_path_dir, dir_name)):
            pass
        else:
            os.mkdir(os.path.join(seg_path_dir, dir_name))

        if os.path.exists(os.path.join(ct_path_dir, dir_name)):
            pass
        else:
            os.mkdir(os.path.join(ct_path_dir, dir_name))

        for i, kidney in enumerate(out):
            final_img = nib.Nifti1Image(kidney[0], ct.affine)
            final_seg = nib.Nifti1Image(kidney[1], seg.affine)
            label = kidney[2]

            seg_save_path = os.path.join(os.path.join(seg_path_dir, dir_name),
                                         f'{case_id}_kidney_{i + 1}_{label}_seg.nii.gz')
            nib.save(final_seg, seg_save_path)
            ct_save_path = os.path.join(os.path.join(ct_path_dir, dir_name), f'{case_id}_kidney_{i + 1}_{label}.nii.gz')
            nib.save(final_img, ct_save_path)


seg_paths1 = glob.glob('/scratch/project_465001979/ct_data/kidney/data/labelsTr/*/*.nii.gz')
seg_paths2 = glob.glob('/scratch/project_465001979/ct_data/kidney/tuh_train/*/labels/*/*.nii.gz')
seg_paths3 = glob.glob('/scratch/project_465001979/ct_data/kidney/tuh_test/*/labels/*/*.nii.gz')

paths = [seg_paths1, seg_paths2, seg_paths3]

for i, p in enumerate(paths):
    print(f"Starting with {i + 1} out of {len(paths)}")
    cropping_pipeline(p)

print("Done!")
