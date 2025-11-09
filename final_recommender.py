from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
import pandas as pd
from scipy import sparse
import google.generativeai as genai

genai.configure(api_key="AIzaSyB66wEZfCyyTI1YPAozhbFuOPBGCq7KPlo")


with open('models/lightfm_model.pkl', 'rb') as f:
    lightFM_model = pickle.load(f)

with open('models/dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

with open('models/movie_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

item_features = sparse.load_npz("models/item_features.npz")
model = SentenceTransformer("intfloat/e5-base-v2")
index = faiss.read_index("models/movie_vector.index")
print("MODELS LOADED")

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
scores = lightFM_model.predict(encoded_user_id, encoded_unseen, item_features=item_features)

top_items = np.argsort(-scores)[:50]
top_item_ids = [unseen_items[i] for i in top_items]

lightFM_recs = metadata[metadata['item_id'].isin(top_item_ids)]

query = "Action / Drama Movies Like Heat"
query_vec = model.encode([query])
D, I = index.search(np.array(query_vec).astype('float32'), k=10)
RAG_recs = metadata.iloc[I[0]]



# Merge semantic and LightFM results
merged = pd.concat([RAG_recs, lightFM_recs]).drop_duplicates('item_id')


context = "\n\n".join(merged['combined_text'].iloc[:5])

prompt = f"""
You are a movie expert.
Based on the following movie details, recommend similar movies.
Context:
{context}
Question: {query}
"""
gemini_model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
response = gemini_model.generate_content(prompt)
print(response.text)