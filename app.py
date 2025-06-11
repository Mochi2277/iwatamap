import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit.components.v1 as components

# --- データ読み込み ---
df = pd.read_csv("iwata_recommend.csv")

st.title("磐田市おすすめスポットレコメンドMAP")

# --- 検索ボックス ---
query = st.text_input("どんな場所を探していますか？（例：子どもと楽しめるカフェ）")

# --- 類似度でフィルタリング ---
if query:
    # TF-IDFベクトル化
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform([query] + df["description"].fillna("").tolist())
    
    # クエリとの類似度計算（index 0 が query）
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    df["similarity"] = similarities
    results = df.sort_values("similarity", ascending=False).head(5)
else:
    results = df.head(5)

# --- カード型の横並びレイアウト（HTML） ---
html_content = """
<style>
.card-container {
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
    justify-content: flex-start;
    margin-top: 20px;
}
.card {
    flex: 1 1 calc(33.333% - 20px);
    background-color: #ffffff;
    border-radius: 10px;
    padding: 16px;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    border: 1px solid #ccc;
    color: #000000;
    max-width: 300px;
}
@media (max-width: 768px) {
    .card {
        flex: 1 1 100%;
    }
}
</style>
<div class="card-container">
"""

for _, row in results.iterrows():
    html_content += f"""
    <div class="card">
        <h4>{row['place_name']}</h4>
        <p><strong>評価:</strong> ⭐️ {row['rating']}（{row['user_ratings_total']}件）</p>
        <p><strong>説明:</strong> {row['description']}</p>
        <p><strong>住所:</strong> {row['formatted_address']}</p>
        <a href="{row['google_map_url']}" target="_blank">📍 Googleマップで開く</a>
    </div>
    """

html_content += "</div>"
components.html(html_content, height=800, scrolling=True)

# --- Googleマイマップ埋め込み（全体地図） ---
st.subheader("🌏 全体マップ（Googleマイマップ）")
st.markdown("""
<iframe src="https://www.google.com/maps/d/u/0/embed?mid=1IhaKYfz5dvbJgV7VPcCjjM7rDchV9tE&ehbc=2E312F"
        width="100%" height="600"></iframe>
""", unsafe_allow_html=True)
