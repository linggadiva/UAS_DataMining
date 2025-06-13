import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# --- Load Data ---
df = pd.read_csv("/DATA/flipkart_com-ecommerce_sample.csv")

# --- Preprocessing ---
df_cluster = df[['harga_diskon', 'rating']].copy()
df_cluster = df_cluster.dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster)

# --- KMeans Clustering ---
kmeans = KMeans(n_clusters=3, random_state=42)
df_cluster['cluster'] = kmeans.fit_predict(X_scaled)

# --- Streamlit UI ---
st.set_page_config(page_title="E-Commerce Analytics", layout="wide")
st.title("📊 Dashboard Produk E-Commerce")

# Sidebar menu
menu = st.sidebar.radio("Pilih Halaman", ["Dashboard Klasterisasi", "Prediksi Best Seller", "Rekomendasi Produk"])

# --- Halaman 1: Klasterisasi Produk ---
if menu == "Dashboard Klasterisasi":
    st.header("🔍 Klasterisasi Produk (KMeans)")
    fig = px.scatter(
        df_cluster, x='harga_diskon', y='rating', color=df_cluster['cluster'].astype(str),
        title="Hasil Klasterisasi Produk",
        labels={'cluster': 'Klaster'},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Deskripsi Klaster:**
    - Klaster 0: Produk harga menengah, rating stabil
    - Klaster 1: Produk murah & laris (rating tinggi)
    - Klaster 2: Produk mahal & kurang diminati (rating rendah)
    """)

# --- Halaman 2: Prediksi Produk Best Seller ---
elif menu == "Prediksi Best Seller":
    st.header("📈 Prediksi Produk Best Seller")
    with st.form("form_prediksi"):
        harga_retail = st.number_input("Harga Retail", min_value=0)
        harga_diskon = st.number_input("Harga Diskon", min_value=0)
        rating = st.slider("Rating Produk", 0.0, 5.0, step=0.1)
        submitted = st.form_submit_button("Prediksi")

    if submitted:
        diskon = harga_retail - harga_diskon
        X_pred = scaler.transform([[harga_diskon, rating]])
        cluster_pred = kmeans.predict(X_pred)[0]

        # Logika prediksi sederhana (contoh)
        if cluster_pred == 1:
            hasil = "✅ Produk ini berpotensi jadi Best Seller!"
        else:
            hasil = "⚠️ Produk ini kurang potensial, coba perbaiki harga/rating."

        st.success(hasil)

# --- Halaman 3: Rekomendasi Produk ---
elif menu == "Rekomendasi Produk":
    st.header("💡 Rekomendasi Produk untuk Promosi")
    df['cluster'] = kmeans.predict(scaler.transform(df[['harga_diskon', 'rating']].fillna(0)))

    rekomendasi = df[df['cluster'] == 1].copy()
    rekomendasi['Insight'] = np.where(
        rekomendasi['rating'] < 4,
        "Naikkan rating agar makin laris",
        "Sudah sesuai klaster best-seller"
    )

    st.dataframe(rekomendasi[['nama_produk', 'harga_diskon', 'rating', 'Insight']].head(50))
