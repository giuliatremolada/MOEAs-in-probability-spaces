import os
os.system('conda install -c conda-forge scikit-surprise')
import pandas as pd
from datetime import datetime
from surprise import KNNBasic
from surprise import Dataset, Reader

def fill_matrix(rating_matrix):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(rating_data[['user_id', 'item_id', 'rating']], reader)
    trainset = data.build_full_trainset()
    algo = KNNBasic(k=8, sim_options={'name': 'cosine', 'user_based': True})
    algo.fit(trainset)

    na_keys = []
    for i in rating_matrix.index:
        for j in rating_matrix.columns:
            if rating_matrix.loc[i, j] == 0:
                na_keys.append([i, j])

    for key in na_keys:
        rating_matrix.loc[key[0], key[1]] = round(algo.predict(uid=key[0], iid=key[1]).est)


rating_data = pd.read_csv("C:/Users/39348/Desktop/u.data", sep='\t', engine='python',
                          names=["user_id", "item_id", "rating", "timestamp"])

rating_data['timestamp'] = [datetime.fromtimestamp(x) for x in rating_data['timestamp']]
print(rating_data)
rating_matrix = rating_data.pivot(
    index='user_id',
    columns='item_id',
    values='rating'
).fillna(0)
fill_matrix(rating_matrix)

import pickle
with open("rating_matrix.pickle", "wb") as f:
    pickle.dump(rating_matrix, f)
print(rating_matrix)