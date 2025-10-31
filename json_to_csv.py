import pandas as pd
import csv
dtypes = {
    "user_id": "int64",
    "item_id": "int64",
    "rating": "float32"
}
chunks = pd.read_json('genome_2021/raw/ratings.json', lines=True, chunksize=500_000)
for chunk in chunks:
    chunk = chunk.astype(dtypes)
    chunk = chunk[["user_id", "item_id", "rating"]]

    # Save or append to smaller file
    chunk.to_csv('ratings_clean.csv', mode='a', header=False, index=False)