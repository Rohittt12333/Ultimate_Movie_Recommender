import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from lightfm.evaluation import precision_at_k, auc_score
import pickle

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

user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

ratings['user_id_enc'] = user_encoder.fit_transform(ratings['user_id'])
ratings['item_id_enc'] = item_encoder.fit_transform(ratings['item_id'])
metadata['item_id_enc'] = item_encoder.fit_transform(metadata['item_id'])
dataset.fit(users=ratings['user_id_enc'].unique(),items=metadata['item_id_enc'].unique())

num_users, num_items = dataset.interactions_shape()
print('Num users: {}, num_items {}.'.format(num_users, num_items))
"""

train_list=[]
test_list=[]
for user, user_df in ratings.groupby('userId'):
    train_list.append(user_df[:-5])
    test_list.append(user_df[-5:])

train = pd.concat(train_list)


test = pd.concat(test_list)

print("Train size:", len(train))
print("Test size:", len(test))
print("Users in train:", train['userId'].nunique())
print("Users in test:", test['userId'].nunique())
(interactions_train, weights_train) = dataset.build_interactions(
    (x['userId_enc'], x['movieId_enc'], x['rating']) for _, x in train.iterrows()
)

(interactions_test, weights_test) = dataset.build_interactions(
    (x['userId_enc'], x['movieId_enc'], x['rating']) for _, x in test.iterrows()
)

model = LightFM(loss='logistic', random_state=42)
model.fit(interactions_train, epochs=20, sample_weight=weights_train,verbose=True)
print("DONE")


print("Train precision: %.2f" % precision_at_k(model, interactions_train, k=10).mean())
print("Test precision:  %.2f" % precision_at_k(model, interactions_test, k=10).mean())

print("Train AUC: %.2f" % auc_score(model, interactions_train).mean())
print("Test AUC:  %.2f" % auc_score(model, interactions_test).mean())

with open('/model/lightfm_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('/model/user_encoder.pkl', 'wb') as f:
    pickle.dump(user_encoder, f)

with open('/model/item_encoder.pkl', 'wb') as f:
    pickle.dump(item_encoder, f)
"""