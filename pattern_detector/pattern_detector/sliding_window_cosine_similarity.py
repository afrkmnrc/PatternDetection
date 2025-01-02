import numpy as np
import pandas as pd
from cosine_similarity_function import compute_cosine_sim
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor



def sliding_window_cosine_similarity(data, pattern,  column_pattern ):

    len_iter = 400 if len(pattern) >= 400 else 200
    bin_parser = 3 if len_iter == 400 else 2

    pattern[column_pattern] = pattern[column_pattern].astype(float)
    pattern.reset_index(drop=True, inplace=True)
    pattern['bin'] = pattern.index // bin_parser
    pattern1 = pattern.groupby('bin').agg({column_pattern: 'mean'}).reset_index()
    pattern1 = np.array(pattern1[column_pattern]).reshape(-1, )

    # Prepare data
    data.reset_index(drop=True, inplace=True)
    data[column_pattern] = data[column_pattern].astype(float)
    data['bin'] = data.index // bin_parser
    data1 = data.groupby('bin').agg({column_pattern: 'mean'}).reset_index()
    data1 = np.array(data1[column_pattern]).reshape(-1, 1)

    window_size = len(pattern1)
    step_size = 2

    similarity_dict = {}


    with ThreadPoolExecutor() as executor:  # Adjust max_workers based on your CPU cores
        futures = [executor.submit(compute_cosine_sim, data1, window_size, len_iter, pattern1, i, j)
                   for i in range(0, len(data1) - window_size, step_size)
                   for j in range(0, len_iter, (len_iter // 40))]
        for future in as_completed(futures):
            i, j, similarity = future.result()
            if i not in similarity_dict:
                similarity_dict[i] = {}
            similarity_dict[i][j] = similarity

    return similarity_dict, pattern1, window_size, bin_parser, len_iter

