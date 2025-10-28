import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from lightfm.evaluation import precision_at_k, auc_score
import pickle

dataset = Dataset()
ratings = pd.read_csv('C:/Users/Rohit/Documents/Movie Recommender/ml-latest-small_dev_2018/ml-latest-small/ratings.csv')
movies = pd.read_csv('C:/Users/Rohit/Documents/Movie Recommender/ml-latest-small_dev_2018/ml-latest-small/movies.csv')
recs= pd.read_csv('C:/Users/Rohit/Documents/Movie Recommender/movielens_ratings.csv')

user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

recs=recs.drop('imdb_id',axis=1)
recs=recs.drop('tmdb_id',axis=1)
recs=recs.drop('average_rating',axis=1)

recs=recs.drop('title',axis=1)
recs['userId']=611
ratings=pd.concat([ratings,recs])

ratings['userId_enc'] = user_encoder.fit_transform(ratings['userId'])
ratings['movieId_enc'] = item_encoder.fit_transform(ratings['movieId'])
print(ratings)
dataset.fit(users=ratings['userId_enc'],items=ratings['movieId_enc'])

num_users, num_items = dataset.interactions_shape()
print('Num users: {}, num_items {}.'.format(num_users, num_items))


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

with open('lightfm_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('user_encoder.pkl', 'wb') as f:
    pickle.dump(user_encoder, f)

with open('item_encoder.pkl', 'wb') as f:
    pickle.dump(item_encoder, f)
