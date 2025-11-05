import pickle
from lightfm import LightFM
import pandas as pd
import numpy as np
from scipy import sparse
from lightfm.evaluation import precision_at_k, auc_score

item_features = sparse.load_npz("models/item_features.npz")

# Load model and dataset
with open('models/lightfm_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

ratings = pd.read_csv('genome_2021/raw/ratings.csv',names=['user_id', 'item_id', 'rating'])
# Get internal mappings
user_id_map, user_feature_map, item_id_map, item_feature_map = dataset.mapping()

# Example user ID (must exist in your dataset)
target_user = 200001
# Get encoded index
if target_user not in user_id_map:
    raise ValueError("User not in dataset!")

encoded_user_id = user_id_map[target_user]
user_rated_items = ratings.loc[ratings['user_id'] == target_user, 'item_id'].values
unseen_items = [iid for iid in item_id_map.keys() if iid not in user_rated_items]

encoded_unseen = [item_id_map[iid] for iid in unseen_items]
scores = model.predict(encoded_user_id, encoded_unseen)

top_items = np.argsort(-scores)[:20]
top_item_ids = [unseen_items[i] for i in top_items]


# Load your metadata
#top_item_ids = [int(i) for i in top_item_ids]
metadata = pd.read_json('genome_2021/raw/metadata.json', lines=True)
recommendations = metadata[metadata['item_id'].isin(top_item_ids)]
print(recommendations[['item_id', 'title']])
