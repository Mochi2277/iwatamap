import streamlit as st
import pandas as pd
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# --- データ読み込み ---
with open("embeddings.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# --- Streamlit UI ---
st.title("磐田市おすすめスポット検索（FastTextベース）")

query = st.text_input("どんな場所を探していますか？（例：自然が多いカフェ）")

def get_query_vector(query_text, model_dim=300):
    # テキストを単語に分割（かなり簡易的）
    words = query_text.split()
    vectors = []
    for word in words:
        for row in df["vector"]:
            if word in row:  # プレースホルダ：すでに埋め込まれてるのでここではゼロベクトル
                vectors.append(row[word])
    if not vectors:
        return np.zeros((1, model_dim))
    return np.mean(vectors, axis=0).reshape(1, -1)

if query:
    # クエリベクトル（簡易的にTF-IDFでもOK）
    vectorizer = TfidfVectorizer()
    all_texts = [x["description"] for x in data]
    vectorizer.fit(all_texts)
    query_vec = vectorizer.transform([query])

    doc_vecs = vectorizer.transform(all_texts)

    similarities = cosine_similarity(query_vec, doc_vecs)[0]
    df["similarity"] = similarities
    top_results = df.sort_values("similarity", ascending=False).head(5)
else:
    top_results = df.head(5)

# --- 検索結果表示（カード型） ---
st.subheader("🔍 おすすめスポット")

for _, row in top_results.iterrows():
    st.markdown(f"""
    <div style="border:1px solid #ccc; border-radius:8px; padding:12px; margin-bottom:16px; background-color:#f9f9f9;">
        <h4>{row['place_name']}</h4>
        <p><strong>評価:</strong> ⭐️ {row['rating']}（{row['user_ratings_total']}件）</p>
        <p><strong>説明:</strong> {row['description']}</p>
        <p><strong>住所:</strong> {row['formatted_address']}</p>
        <a href="{row['google_map_url']}" target="_blank">📍 Googleマップで開く</a>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")

# --- 結果をfoliumマップで表示するのも可能（必要なら追記） ---
