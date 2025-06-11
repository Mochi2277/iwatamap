import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from sentence_transformers import SentenceTransformer, util

# --- データ読み込み ---
df = pd.read_csv("iwata_recommend.csv")
df["description"] = df["description"].fillna("")

# --- モデル読み込み（日本語BERTモデル） ---
model = SentenceTransformer("sonoisa/sentence-bert-base-ja-mean-tokens")

# --- 事前に文ベクトルを計算（重いのでキャッシュ） ---
@st.cache_resource
def compute_embeddings(texts):
    return model.encode(texts, convert_to_tensor=True)

embeddings = compute_embeddings(df["description"].tolist())

# --- ユーザー入力（検索クエリ） ---
query = st.text_input("どんな場所を探していますか？（例：子連れ カフェ）")
if query:
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, embeddings)[0].cpu().numpy()
    df["similarity"] = similarities
    results = df.sort_values("similarity", ascending=False).head(5)
else:
    results = df.head(5)

# --- 検索結果表示（カード形式） ---
st.subheader("おすすめスポット")
for _, row in results.iterrows():
    st.markdown(f"""
    <div style="border:1px solid #ccc; padding:12px; margin:8px 0; border-radius:8px;">
    <b>{row['place_name']}</b>（評価: ⭐️{row['rating']} / {row['user_ratings_total']}件）<br>
    {row['description']}<br>
    📍 <a href="{row['google_map_url']}" target="_blank">Googleマップで見る</a>
    </div>
    """, unsafe_allow_html=True)

# --- 地図表示 ---
st.subheader("地図で表示")
m = folium.Map(location=[34.7, 137.85], zoom_start=12)
for _, row in results.iterrows():
    folium.Marker(
        [row["geometry_lat"], row["geometry_lng"]],
        popup=row["place_name"]
    ).add_to(m)
st_folium(m, width=700, height=500)
