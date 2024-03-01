import os
from time import time
import pandas as pd
from sentence_transformers import SentenceTransformer
from itertools import groupby
from operator import itemgetter
from src.start import tokenizer, vectorizer
from sklearn.cluster import AgglomerativeClustering


def grouped_func(data: list) -> [{}]:
    """Function groups input list of data with format: [(label, vector, text)]
    into list of dictionaries, each dictionary of type:
    {
    label: label,
    texts: list of texts correspond to label
    vectors_matrix: numpy matrix of vectors correspond to label
    }
    """
    data = sorted(data, key=lambda x: x[0])
    grouped_data = []
    for key, group_items in groupby(data, key=itemgetter(0)):
        d = {"label": key, "texts": []}
        for item in group_items:
            d["texts"].append(item[1])
        grouped_data.append(d)
    return grouped_data



def clustering_func(vectorizer: SentenceTransformer, clusterer: AgglomerativeClustering, texts: []) -> {}:
    """Function for text collection clustering"""
    vectors = vectorizer.encode([x.lower() for x in texts])
    clusters = clusterer.fit(vectors)
    # data = [(lb, v, tx) for lb, v, tx in zip(clusters.labels_, vectors, texts)]
    data = [(lb, tx) for lb, v, tx in zip(clusters.labels_, vectors, texts)]
    grouped_data = grouped_func(data)
    result_list = []
    for d in grouped_data:
        label = str(d["label"])
        cluster_size = len(d["texts"])
        result_list += [{"cluster_num": label, "lem_request_string": tx, "cluster_size": cluster_size} for tx in d["texts"]]
    return result_list

clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=1.0, memory=os.path.join("cache"))


chunk_size = 500000
for fn in ["hs_test_data_for_reform_recognition.csv"]:
    path = os.path.join(os.getcwd(), "data", fn)
    df_iter = pd.read_csv(path, sep="\t", encoding = "utf-16", on_bad_lines='skip', chunksize=chunk_size)
    for k, df in enumerate(df_iter):
        df["serverTimestamp"] = df["serverTimestamp"].astype("datetime64[ns]")
        
        dates = list(set(df["serverTimestamp"]))
        texts = list(df["payload__request_string"])
        
        # добавим лемматизированные тексты:
        lm_texts_df = pd.DataFrame([{"lem_request_string": " ".join(lm_tx)} for lm_tx in tokenizer(texts)])
        df = pd.concat([df, lm_texts_df], axis=1)
        
        result_dfs = []
        for num, date in enumerate(dates):
            t = time()
            
            df_date = df[df["serverTimestamp"] == date]
            df_tms_grp = df_date[["new_licensesId", "serverTimestamp"]].groupby("new_licensesId", as_index=False).count()
            users_more = list(set(df_tms_grp["new_licensesId"][df_tms_grp["serverTimestamp"] > 1])) # пользователи у которых больше 1ого сообщения
            
            temp_result = []
            for user in users_more:
                try:
                    df_date_user = df_date[df_date["new_licensesId"] == user]
                    lm_texts = list(df_date_user["lem_request_string"])
                    clustering_dicts_df = pd.DataFrame(clustering_func(vectorizer, clusterer, lm_texts))
                    temp_clustering_user_df = pd.merge(df_date_user, clustering_dicts_df, on="lem_request_string")
                    result_dfs.append(temp_clustering_user_df)
                except:
                    pass
            
            users_one = list(set(df_tms_grp["new_licensesId"][df_tms_grp["serverTimestamp"] == 1]))
            if users_one:
                try:
                    df_date_one = df_date[df_date["new_licensesId"].isin(users_one)]
                    temp_one_df = pd.DataFrame([{**d, **{"cluster_num": 0, "cluster_size": 1}}  for d in df_date_one.to_dict(orient="records")])
                    result_dfs.append(temp_one_df)
                except:
                    pass
            print(num + 1, "/", len(dates), "working time, sec:", time() - t)
        result_df = pd.concat(result_dfs, axis=0)
        result_df.drop("lem_request_string", axis=1).to_csv(os.path.join(os.getcwd(), "results", str(k) + "_" + fn), index=False, sep="\t")