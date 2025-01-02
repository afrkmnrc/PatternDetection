import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.fft import fft
from scipy.interpolate import interp1d


def compute_cosine_sim(data1, window_size,len_iter,pattern1,  i, j):

    window = data1[i:i + window_size - (len_iter//2) + j ,:]


    max_pos = np.max(pattern1) +  0.1 * np.ptp(pattern1) # np.ptp : calculates max-min difference
    min_pos = np.min(pattern1) - 0.1 * np.ptp(pattern1)
    mean_pos_upper = np.mean(pattern1) + 0.1 * np.ptp(pattern1)
    mean_pos_lower = np.mean(pattern1) - 0.1 * np.ptp(pattern1)
    pattern_skewness = skew(pattern1.flatten())
    pattern_kurtosis = kurtosis(pattern1.flatten())
    pattern_std = np.std( pattern1.flatten() )
    cross_correlation_threshold = 0.5  # Set a threshold for cross-correlation

    starting_point_lower = pattern1[0] - 0.2 * np.ptp(pattern1)
    starting_point_upper = pattern1[0] + 0.2 * np.ptp(pattern1)

    ending_point_lower = pattern1[-1] - 0.2 * np.ptp(pattern1)
    ending_point_upper = pattern1[-1] + 0.2 * np.ptp(pattern1)


    x_original = np.linspace(0, 1, len(window))
    x_target = np.linspace(0, 1, len(pattern1))

    window = window.reshape(-1 ,)

    interpolator = interp1d(x_original, window, kind='cubic')  # Linear interpolation

    window2 = interpolator(x_target)

    corr_coef = np.corrcoef(window2, pattern1)[0][1]

    sliding_window_max = np.max(window2)
    sliding_window_min = np.min(window2)
    sliding_window_skewness = skew(window2)
    sliding_window_kurtosis = kurtosis(window2)
    sliding_window_mean = np.mean(window2)
    sliding_window_std = np.std(window2)


    if ( sliding_window_max <= max_pos and sliding_window_min >= min_pos and
            mean_pos_upper >= sliding_window_mean >= mean_pos_lower and
            abs(sliding_window_skewness - pattern_skewness) < 0.5 and
            abs(sliding_window_kurtosis - pattern_kurtosis) < 1.0 and
            pattern_std * 0.9 <= sliding_window_std <= pattern_std * 1.1 and
            corr_coef >= cross_correlation_threshold
            and starting_point_lower<= window2[0] <= starting_point_upper and
            ending_point_lower<= window2[-1] <= ending_point_upper ):

        fft_pattern = fft(pattern1)
        fft_window = fft(window2)

        magnitude_pattern = np.abs(fft_pattern)
        magnitude_window = np.abs(fft_window)

        dot_product = np.dot(magnitude_pattern, magnitude_window)
        norm_1 = np.linalg.norm(magnitude_pattern)
        norm_2 = np.linalg.norm(magnitude_window)

        cosine_similarity = dot_product / (norm_1 * norm_2)


        return i, j, cosine_similarity

    else:
        return i, j, 0
