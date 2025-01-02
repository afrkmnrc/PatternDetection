import numpy as np
import pandas as pd
from pattern_detector.sliding_window_cosine_similarity import  sliding_window_cosine_similarity


def run_area_of_interest_finder(df,pattern,column_pattern):

    similarity_dict, pattern1, window_size, bin_parser, len_iter  = sliding_window_cosine_similarity(df, pattern, column_pattern)

    approx_cycle_length = len(pattern1)*0.95

    results = []
    for key1, value in similarity_dict.items():

        max_key = max(value, key=value.get)
        max_value = value[max_key]
        results.append({'key': key1, 'max_key': max_key, 'max_value': max_value})

    df_dist = pd.DataFrame(results)

    df_dist.reset_index(inplace=True)
    df_dist['app_cycle'] = df_dist["key"] // approx_cycle_length
    df_dist["app_cycle"] = df_dist["app_cycle"].astype(int)

    yig = tuple(df_dist.groupby("app_cycle"))
    cyc_dict = {x: y for x, y in yig}

    idx_cyc = 0
    cyc_concat_df = pd.DataFrame()

    for k in cyc_dict.keys():
        df_cyc = cyc_dict[k]
        df_cyc = df_cyc[ df_cyc["max_value"] != 0 ]

        key_min_df =  df_cyc[["key","max_key","max_value"]][  df_cyc["max_value"] == np.max(df_cyc['max_value'])]
        key_min_df["cycle"] = idx_cyc
        if len(key_min_df) != 0:
            cyc_concat_df = pd.concat([cyc_concat_df,key_min_df],ignore_index=True,axis="index")
            idx_cyc += 1
        else:
            continue

    cyc_concat_df["start_index"] = cyc_concat_df["key"]
    cyc_concat_df["end_index"] = cyc_concat_df["start_index"] + window_size + cyc_concat_df["max_key"] - (len_iter//2)
    cyc_concat_df["shift_start"] = cyc_concat_df["start_index"].shift(1)

    cyc_concat_df["diff"] = cyc_concat_df["shift_start"] - cyc_concat_df["start_index"]
    cyc_concat_df["shift_start"].iloc[0] = len(pattern1)
    cyc_concat_df["diff"].iloc[0] = -len(pattern1)
    limit = len(pattern1)*.7
    cyc_concat_df = cyc_concat_df[ cyc_concat_df["diff"] < -limit ]
    cyc_concat_df.reset_index(inplace=True, drop=True)
    cyc_concat_df["cycle"] = cyc_concat_df.index

    cyc_concat_df["shift_end"] = cyc_concat_df["end_index"].shift(1)
    ######## Çakışmaları önlemek için yapıldı
    cyc_concat_df["shift_end"].iloc[0] = cyc_concat_df["diff"].iloc[0]
    cyc_concat_df["diff_end"] = cyc_concat_df["shift_end"] - cyc_concat_df["start_index"]
    cyc_concat_df["start_index"][ cyc_concat_df["diff_end"] > 0 ] =  cyc_concat_df["start_index"][ cyc_concat_df["diff_end"] > 0 ] + cyc_concat_df["diff_end"] + 1


    #df = data.copy()
    df.reset_index(drop=True,inplace=True)
    for i in cyc_concat_df["cycle"].unique():
        start = cyc_concat_df["start_index"][cyc_concat_df["cycle"] == i].values[0]*bin_parser
        stop = cyc_concat_df["end_index"][cyc_concat_df["cycle"] == i].values[0]*bin_parser
        #print(start, stop, i, stop-start)
        df.loc[start:stop,"cycle"] = int(i)


    return df
