from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pandas as pd

metadata = pd.read_json('genome_2021/raw/review_data.json', lines=True)
model = SentenceTransformer("intfloat/e5-base-v2")

embeddings = model.encode(metadata['combined_text'].tolist(),normalize_embeddings=True,batch_size=64, show_progress_bar=True)
embeddings = np.array(embeddings).astype('float32')

# Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, "models/movie_vector.index")
metadata.to_pickle("models/movie_metadata.pkl")

