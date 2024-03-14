import os
import pandas as pd

s = 0
for k in range(20):
    fn = str(k) + "_" + "hs_ss_1020_1022_search_str_all_asis_by_day.csv"
    temp_df = pd.read_csv(os.path.join(os.getcwd(), "results", "240312", fn), sep="\t")
    print(k, temp_df.shape)
    s += temp_df.shape[0]

print(s)