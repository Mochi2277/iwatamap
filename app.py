import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

# --- データ読み込み ---
df = pd.read_csv("iwata_recommend.csv")

st.title("磐田市おすすめスポットレコメンドMAP")

# --- 検索ボックス ---
query = st.text_input("どんな場所を探していますか？（例：子どもと楽しめるカフェ）")

# --- 検索結果フィルタリング ---
if query:
    results = df[df["description"].str.contains(query, case=False, na=False)].head(5)
else:
    results = df.head(5)

# --- 検索結果のテーブル表示 ---
st.subheader("🔍 検索結果")
st.dataframe(results[["place_name", "description", "formatted_address", "rating", "user_ratings_total"]])

# --- Foliumで検索結果だけの地図を表示 ---
st.subheader("🗺 結果スポットのミニマップ（検索結果のみ）")
m = folium.Map(location=[34.7, 137.85], zoom_start=12)
for _, row in results.iterrows():
    folium.Marker(
        location=[row["geometry_lat"], row["geometry_lng"]],
        popup=f"<b>{row['place_name']}</b><br>{row['description']}",
        tooltip=row["place_name"],
    ).add_to(m)
st_folium(m, width=700, height=500)

# --- Googleマイマップ埋め込み（全体ビュー） ---
st.subheader("🌏 全体マップ（Googleマイマップ）")
st.markdown("""
<iframe src="https://www.google.com/maps/d/u/0/embed?mid=1IhaKYfz5dvbJgV7VPcCjjM7rDchV9tE&ehbc=2E312F" 
        width="100%" height="600"></iframe>
""", unsafe_allow_html=True)
