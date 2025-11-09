#PROGRAM TO WRITE JSON DATASET TO CSV

import pandas as pd
import csv

chunks = pd.read_json('genome_2021/raw/reviews.json', lines=True, chunksize=500_000)
for chunk in chunks:

    chunk = chunk[["item_id", "txt"]]

    # Save or append to smaller file
    chunk.to_csv('reviews.csv', mode='a', header=False, index=False)