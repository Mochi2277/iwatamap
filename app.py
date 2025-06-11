import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

# データ読み込み
df = pd.read_csv("iwata_recommend.csv")

st.title("磐田市おすすめスポットレコメンドMAP")

# 検索ボックス
query = st.text_input("どんな場所を探していますか？（例：子どもと楽しめるカフェ）")

# フィルター処理（説明文に含まれるかで）
if query:
    results = df[df["description"].str.contains(query, case=False, na=False)].head(5)
else:
    results = df.head(5)

# 結果表示
st.subheader("おすすめスポット")
st.dataframe(results[["place_name", "description", "formatted_address", "rating", "user_ratings_total"]])

# 地図生成
m = folium.Map(location=[34.7, 137.85], zoom_start=12)
for _, row in results.iterrows():
    folium.Marker(
        location=[row["geometry_lat"], row["geometry_lng"]],
        popup=f"<b>{row['place_name']}</b><br>{row['description']}",
        tooltip=row["place_name"],
    ).add_to(m)

st_folium(m, width=700, height=500)
