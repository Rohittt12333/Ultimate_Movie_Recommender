# Ultimate Movie Recommender System  

A **Hybrid AI-powered Movie Recommendation System** that combines **Collaborative Filtering**, **Content-Based Filtering**, and **Semantic Search (RAG)** for personalized movie suggestions.  Trained using MovieLens Genome 2021 dataset.

---

## Overview  
This project integrates multiple AI techniques to recommend movies tailored to user preferences and query context:  
- **LightFM Model:** Learns from user–movie ratings using both collaborative and content-based signals.  
- **Semantic Retrieval (RAG):** Uses `SentenceTransformer` embeddings and a `FAISS` vector index for meaning-based similarity search.  
- **Hybrid Ranking:** Merges LightFM predictions and semantic results, refined through Gemini API for natural-language reasoning.
  

---

## Technical Features  
- **Frameworks:** LightFM, Sentence Transformers, FAISS, Google Gemini API  
- **Hybrid Architecture:** Combines interaction-based and context-based recommendation  
- **Metadata Integration:** Includes tags, actors, directors, and average ratings for content representation  
- **Semantic Query Support:** Handles natural queries like *“movies like Inception”* or *“gritty psychological thrillers”*  
- **Vector Search:** Efficient nearest-neighbor retrieval using FAISS index  
- **Extensible Design:** Easily expandable with new metadata or embeddings  

---

## Core Components  
- `LightFM` — Collaborative + Content-based model  
- `SentenceTransformer (e5-base-v2)` — Text embedding for movie metadata  
- `FAISS` — Fast similarity search over vectorized movie data  
- `Gemini API` — Natural-language refinement and explanation  


---

## Outcome  
A unified recommendation pipeline that provides:  
- Personalized movie suggestions  
- Context-aware semantic search  
- Natural-language explanation of recommendations  

---

## Technologies Used  
`Python`, `LightFM`, `SentenceTransformers`, `FAISS`, `Google Generative AI`, `Pandas`, `Scipy`, `Streamlit`

---
