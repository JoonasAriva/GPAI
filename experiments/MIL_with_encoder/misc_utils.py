from __future__ import print_function

import random
import re

import numpy as np
from sklearn.metrics import average_precision_score


# def attention_accuracy(attention, df):
#     df.loc[(df["kidney"] > 0) | (df["tumor"] > 0) | (df["cyst"] > 0), "important_all"] = 1
#     df["important_all"] = df["important_all"].fillna(0)
#
#     df.loc[df["tumor"] > 0, "important_tumor"] = 1
#     df["important_tumor"] = df["important_tumor"].fillna(0)
#
#     # print("attention min, max and nan values: ",np.min(attention),np.max(attention),np.isnan(attention).any())
#     treshold = threshold_otsu(attention)
#     # otsu threshold
#     # binarize attention
#     attention[attention > treshold] = 1
#     attention[attention != 1] = 0
#
#     matching_all = np.sum((attention == True) & (np.array(df["important_all"] == True)))
#     matching_tumor = np.sum((attention == True) & (np.array(df["important_tumor"] == True)))
#     all_relevant_attention = np.sum(attention)
#     all_relevant_slices = np.sum(df["important_all"])
#
#     accuracy_all = matching_all / all_relevant_attention
#     recall_all = matching_all / all_relevant_slices
#     accuracy_tumor = matching_tumor / all_relevant_attention
#
#     return round(accuracy_all, 2), round(accuracy_tumor, 2), round(recall_all, 2)


def find_case_id(path, start_string, end_string):
    match = re.search(start_string + '(.*)' + end_string, path[0]).group(1)
    match = match.strip("_0000")
    return match


def center_crop_dataframe(df, crop_size):
    if len(df) > crop_size:
        # simulate center cropper
        midpoint = int(len(df) / 2)
        df = df[int(midpoint - crop_size / 2):int(midpoint + crop_size / 2)].copy()

        df.reset_index(inplace=True)
    return df


def prepare_statistics_dataframe(df, case_id, crop_size, nth_slice, roll_slices):
    scan_df = df.loc[df["file_name"] == case_id]

    scan_df = scan_df.copy()[::nth_slice]
    # print("len after nth slice sampling: ", len(scan_df))
    cropped_scan_df = center_crop_dataframe(scan_df, crop_size)
    if roll_slices:
        cropped_scan_df = cropped_scan_df[1:-1].copy()
    return cropped_scan_df


def evaluate_attention(attention, df, case_id, crop_size, nth_slice, bag_label, roll_slices):
    cropped_scan_df = prepare_statistics_dataframe(df, case_id, crop_size, nth_slice, roll_slices)
    ap_all, ap_tumor = calculate_ap(attention, cropped_scan_df, bag_label)

    return ap_all, ap_tumor


def calculate_ap(attention, df, bag_label):
    df.loc[(df["kidney"] > 0) | (df["tumor"] > 0) | (df["cyst"] > 0), "important_all"] = 1
    df["important_all"] = df["important_all"].fillna(0)

    df.loc[df["tumor"] > 0, "important_tumor"] = 1
    df["important_tumor"] = df["important_tumor"].fillna(0)

    attention = attention.detach().cpu().numpy()

    ap_all = average_precision_score(df["important_all"], attention)

    if bag_label:
        ap_tumor = average_precision_score(df["important_tumor"], attention)
    else:
        ap_tumor = 0
    return round(ap_all, 2), round(ap_tumor, 2)


def get_percentage_of_scans_from_dataframe(df, percentage):
    grouped_stats = df.groupby("file_name").sum()
    no_tumor = grouped_stats[grouped_stats["tumor"] == 0].reset_index().file_name.unique()
    tumor = grouped_stats[grouped_stats["tumor"] > 0].reset_index().file_name.unique()
    random.shuffle(tumor)
    random.shuffle(no_tumor)
    tumor = tumor[:int(percentage * len(tumor))]
    no_tumor = no_tumor[:int(percentage * len(no_tumor))]

    return np.concatenate([tumor, no_tumor])
