import numpy as np
import pandas as pd
def load_toydata(num_series, seq_length):
    data_list = []
    for i in range(num_series):
        x = np.arange(seq_length)
        level = 10 ** (i % 3) 
        y = level * np.sin(x / 5.0) + np.random.normal(scale=0.1, size=seq_length)
        category = i % 5
        static_value = np.random.rand()
        for t in range(seq_length - 1):
            data_list.append(
                {
                    "series_id": i,
                    "time_idx": t,
                    "x": y[t],
                    "y": y[t + 1],
                    "category": category,
                    "future_known_feature": np.cos(t / 10),
                    "static_feature": static_value,
                    "static_feature_cat": i % 3,
                }
            )
    data_df = pd.DataFrame(data_list)
    return data_df