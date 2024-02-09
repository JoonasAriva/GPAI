import re

import numpy as np
from skimage.filters import threshold_otsu


def attention_accuracy(attention, df):

    df.loc[(df["kidney"] > 0) | (df["tumor"] > 0) | (df["cyst"] > 0), "important_all"] = 1
    df["important_all"] = df["important_all"].fillna(0)

    df.loc[df["tumor"] > 0, "important_tumor"] = 1
    df["important_tumor"] = df["important_tumor"].fillna(0)

    #print("attention min, max and nan values: ",np.min(attention),np.max(attention),np.isnan(attention).any())
    treshold = threshold_otsu(attention)
    # otsu threshold
    # binarize attention
    attention[attention > treshold] = 1
    attention[attention != 1] = 0

    matching_all = np.sum((attention == True) & (np.array(df["important_all"] == True)))
    matching_tumor = np.sum((attention == True) & (np.array(df["important_tumor"] == True)))
    all_relevant_attention = np.sum(attention)
    all_relevant_slices = np.sum(df["important_all"])


    accuracy_all = matching_all / all_relevant_attention
    recall_all = matching_all / all_relevant_slices
    accuracy_tumor = matching_tumor / all_relevant_attention

    return round(accuracy_all, 2), round(accuracy_tumor, 2), round(recall_all, 2)


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
