import pickle
from lightfm import LightFM
import pandas as pd
import numpy as np

# Load model
with open('lightfm_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load encoders
with open('user_encoder.pkl', 'rb') as f:
    user_encoder = pickle.load(f)

with open('item_encoder.pkl', 'rb') as f:
    item_encoder = pickle.load(f)

movies = pd.read_csv('C:/Users/Rohit/Documents/Movie Recommender/ml-latest-small_dev_2018/ml-latest-small/movies.csv')

n_items = np.arange(len(item_encoder.classes_))


scores = model.predict(610, n_items)
top_items = np.argsort(-scores)[:10]
# Decode movie IDs back to original
top_movie_ids = item_encoder.inverse_transform(top_items)
recommendations = movies[movies['movieId'].isin(top_movie_ids)]
print( recommendations[['movieId', 'title']])
movies = pd.read_csv('C:/Users/Rohit/Documents/Movie Recommender/ml-latest-small_dev_2018/ml-latest-small/movies.csv')