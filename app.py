import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors

# --- 準備 ---
df = pd.read_csv("iwata_recommend_base.csv")
embeddings = np.load("spot_embeddings.npy")
model = KeyedVectors.load_word2vec_format("cc.ja.300.vec.gz", binary=False)

# --- 類似検索 ---
query = st.text_input("探している場所は？")
if query:
    def sentence_vector(text):
        words = text.split()
        vecs = [model[word] for word in words if word in model]
        return np.mean(vecs, axis=0).reshape(1, -1) if vecs else np.zeros((1, 300))

    query_vec = sentence_vector(query)
    sims = cosine_similarity(query_vec, embeddings)[0]
    top5_idx = sims.argsort()[::-1][:5]

    st.subheader("おすすめスポット")
    for i in top5_idx:
        st.markdown(f"- **{df.loc[i, 'place_name']}**: {df.loc[i, 'description']}")

