import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
from scipy import sparse

dataset = Dataset()

ratings = pd.read_csv('genome_2021/raw/ratings.csv',names=['user_id', 'item_id', 'rating'])

metadata = pd.read_json('genome_2021/raw/metadata.json',lines=True)

tags = pd.read_json('genome_2021/raw/tags.json', lines=True)

tag_count = pd.read_json('genome_2021/raw/tag_count.json', lines=True)
print("LOADED DATASETS")
#Dropping tags with only 1 use 
tag_count.drop(tag_count.index[(tag_count["num"] == 1)],axis=0,inplace=True)

# Merge tag info 
tag_count = tag_count.merge(tags, left_on='tag_id', right_on='id', how='left')
movie_tags = tag_count.groupby('item_id')['tag'].apply(list).reset_index()
movie_tags.columns = ['item_id', 'tags']

metadata = metadata.merge(movie_tags, on='item_id', how='left')

metadata['tags'] = metadata['tags'].apply(lambda x: x if isinstance(x, list) else [])
print("MERGED DATA")
def combine_features(row):
    features = []
    if 'directedBy' in row and row['directedBy']:
        features.append(f"director:{row['directedBy']}")
    if 'starring' in row and row['starring']:
        for actor in row['starring'].split(','):
            features.append(f"actor:{actor.strip()}")
    if 'avgRating' in row and not pd.isna(row['avgRating']):
        features.append(f"rating_bin:{int(row['avgRating'])}")
    if row.get('tags'):
        for tag in row['tags']:
            features.append(f"tag:{tag.strip().replace(' ', '_')}")
    return features

metadata['features'] = metadata.apply(combine_features, axis=1)
print("COMBINED FEATURES")


all_item_ids = pd.concat([ratings['item_id'], metadata['item_id']]).unique()

dataset.fit(users=ratings['user_id'].unique(),items=all_item_ids,item_features=set(f for feats in metadata['features'] for f in feats))

num_users, num_items = dataset.interactions_shape()
print('Num users: {}, num_items {}.'.format(num_users, num_items))


train_list=[]
test_list=[]

for user, user_df in ratings.groupby('user_id'):
    train_list.append(user_df[:-5])
    test_list.append(user_df[-5:])

train = pd.concat(train_list)

test = pd.concat(test_list)

print("Train size:", len(train))
print("Test size:", len(test))
print("Users in train:", train['user_id'].nunique())
print("Users in test:", test['user_id'].nunique())

(interactions_train, weights_train) = dataset.build_interactions(
    (x['user_id'], x['item_id'], x['rating']) for _, x in train.iterrows()
)

(interactions_test, weights_test) = dataset.build_interactions(
    (x['user_id'], x['item_id'], x['rating']) for _, x in test.iterrows()
)

item_features = dataset.build_item_features(
    (x['item_id'], x['features']) for _, x in metadata.iterrows()
)

print("BUILT INTERACTION AND FEATURE SETS")
print("TRAINING")
model = LightFM(loss='logistic', random_state=42, no_components=75,)
model.fit(interactions_train, item_features=item_features, epochs=20, sample_weight=weights_train,verbose=True)
print("TRAINING DONE")

sparse.save_npz("models/item_features.npz", item_features)
with open('models/lightfm_model.pkl', 'wb') as f:
    pickle.dump(model, f)


print("MODEL SAVED")
"""
print("Train precision: %.2f" % precision_at_k(model, interactions_train,item_features=item_features, k=10).mean())
print("Test precision:  %.2f" % precision_at_k(model, interactions_test,item_features=item_features, k=10).mean())

print("Train AUC: %.2f" % auc_score(model, interactions_train,item_features=item_features).mean())
print("Test AUC:  %.2f" % auc_score(model, interactions_test,item_features=item_features).mean())

"""