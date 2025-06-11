import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# データ読み込み
df = pd.read_csv("iwata_recommend.csv")

# 入力
query = st.text_input("どんな気分ですか？（例：レトロなカフェ・のんびりしたい）")

if query:
    # 類似度計算
    vec = TfidfVectorizer()
    tfidf_matrix = vec.fit_transform(df["description"].fillna(""))
    query_vec = vec.transform([query])
    sims = cosine_similarity(query_vec, tfidf_matrix)[0]
    df["similarity"] = sims
    results = df.sort_values("similarity", ascending=False).head(5)
else:
    results = df.head(5)

# 表示（カード＋リンク）
st.subheader("おすすめスポット")
for _, row in results.iterrows():
    st.markdown(f"""
    <div style="border:1px solid #ccc; padding:12px; margin:8px 0; border-radius:8px;">
    <b>{row['place_name']}</b>（評価: ⭐️{row['rating']} / {row['user_ratings_total']}件）<br>
    {row['description']}<br>
    📍 <a href="{row['google_map_url']}" target="_blank">Googleマップで見る</a>
    </div>
    """, unsafe_allow_html=True)

# 地図で表示
st.subheader("地図で表示")
m = folium.Map(location=[34.7, 137.85], zoom_start=12)
for _, row in results.iterrows():
    folium.Marker(
        [row["geometry_lat"], row["geometry_lng"]],
        popup=row["place_name"]
    ).add_to(m)
st_folium(m, width=700, height=500)
