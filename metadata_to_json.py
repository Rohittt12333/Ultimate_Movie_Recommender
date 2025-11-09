#PROGRAM TO EXTRACT REVIEWS AND METADATA AND WRITE TO A JSON FILE

import pandas as pd
import json

metadata = pd.read_json('genome_2021/raw/metadata.json', lines=True)
reviews = pd.read_json('genome_2021/raw/reviews.json', lines=True)
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
# Optionally aggregate reviews per movie
review_texts = reviews.groupby('item_id')['txt'].apply(lambda x: ' '.join(x[:10]))  # limit to 10 reviews
metadata = metadata.merge(review_texts, on='item_id', how='left')

metadata['combined_text'] = (
    "Title: " + metadata['title'] + ". " +
    "Directed by: " + metadata['directedBy'].fillna('Unknown') + ". " +
    "Starring: " + metadata['starring'].fillna('Unknown') + ". " +
    "Tags: " + metadata.get('tags', '').astype(str) + ". " +
    "Average Rating: " + metadata['avgRating'].astype(str) + ". " +
    "Reviews: " + metadata['txt'].fillna('')
)
metadata[['item_id','combined_text']].to_json('data.json', orient='records', lines=True)