import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# --- データ読み込み ---
df = pd.read_csv("iwata_recommend.csv")

st.title("磐田市おすすめスポットレコメンドMAP")

# --- 検索ボックス ---
query = st.text_input("どんな場所を探していますか？（例：子どもと楽しめるカフェ）")

if query:
    # TF-IDFベクトル化
    corpus = df["description"].fillna("").tolist()
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus + [query])
    
    # 類似度計算
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
    
    # 類似度上位5件のインデックスを取得
    top_indices = cosine_sim.argsort()[-5:][::-1]
    top_results = df.iloc[top_indices].copy()
    top_results["similarity"] = cosine_sim[top_indices]

    # --- CSSで横並び ---
    st.markdown("""
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
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        border: 1px solid #ccc;
        color: #000000; /* ← ここが重要！文字色を黒に */
        max-width: 300px;
    }
    @media (max-width: 768px) {
        .card {
            flex: 1 1 100%;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card-container">', unsafe_allow_html=True)

    # --- カードごとの出力 ---
    for _, row in top_results.iterrows():
        st.markdown(f"""
        <div class="card">
            <h4>{row['place_name']}</h4>
            <p><strong>評価:</strong> ⭐️ {row['rating']}（{row['user_ratings_total']}件）</p>
            <p><strong>説明:</strong> {row['description']}</p>
            <p><strong>住所:</strong> {row['formatted_address']}</p>
            <p><strong>類似度:</strong> {row['similarity']:.3f}</p>
            <a href="{row['google_map_url']}" target="_blank">📍 Googleマップで開く</a>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
