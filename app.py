import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# =====================
# Load Model dan Data
# =====================
@st.cache_data
def load_models():
    with open("random_forest_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_models()

@st.cache_data
def load_cluster_data():
    return pd.read_csv("produk_klaster.csv")  # Harus punya kolom: nama_produk, harga, rating, ulasan, kategori, cluster

df_clustered = load_cluster_data()

# =====================
# Sidebar Navigasi
# =====================
st.sidebar.title("🧭 Navigasi")
page = st.sidebar.radio("Menu", ["📊 Dashboard Klaster", "🤖 Prediksi Produk", "📌 Rekomendasi Produk"])

# =====================
# 1. Dashboard Klaster
# =====================
if page == "📊 Dashboard Klaster":
    st.title("📊 Dashboard Klasterisasi Produk")

    fig = px.scatter(df_clustered, x="harga", y="rating", color="cluster",
                     hover_data=["nama_produk", "kategori"],
                     title="Visualisasi Klaster Produk")
    st.plotly_chart(fig)

    st.subheader("Deskripsi Klaster:")
    deskripsi = {
        0: "💎 Klaster 0: Produk premium, harga tinggi, rating tinggi",
        1: "🔥 Klaster 1: Produk murah dan laris, ulasan banyak",
        2: "🌱 Klaster 2: Produk baru atau kurang populer",
    }
    for cid, desc in deskripsi.items():
        st.markdown(f"- **Klaster {cid}**: {desc}")

# ==========================
# 2. Prediksi Produk
# ==========================
elif page == "🤖 Prediksi Produk":
    st.title("🤖 Prediksi Produk Best Seller")

    nama = st.text_input("Nama Produk")
    harga = st.number_input("Harga Produk", 0.0)
    rating = st.slider("Rating (0-5)", 0.0, 5.0, 4.0)
    ulasan = st.number_input("Jumlah Ulasan", 0)
    kategori = st.selectbox("Kategori", sorted(df_clustered["kategori"].unique()))

    if st.button("Prediksi"):
        # Pastikan kategori masuk dalam fitur (jika modelnya pakai kategori string)
        fitur = pd.DataFrame([[harga, rating, ulasan, kategori]],
                             columns=["harga", "rating", "ulasan", "kategori"])

        try:
            fitur_scaled = scaler.transform(fitur.select_dtypes(include=[np.number]))
            fitur_scaled_df = pd.DataFrame(fitur_scaled, columns=["harga", "rating", "ulasan"])
            fitur_scaled_df["kategori"] = fitur["kategori"].values

            pred = model.predict(fitur_scaled_df)[0]

            if pred == 1:
                st.success(f"✅ Produk **'{nama}'** diprediksi berpotensi menjadi **Best Seller!**")
            else:
                st.warning(f"⚠️ Produk **'{nama}'** diprediksi **bukan** best seller.")
        except Exception as e:
            st.error(f"Gagal memproses input: {e}")

# ==========================
# 3. Rekomendasi Produk
# ==========================
elif page == "📌 Rekomendasi Produk":
    st.title("📌 Rekomendasi Produk Promosi")

    klaster_target = 1  # Klaster produk laris
    rekomendasi = df_clustered[df_clustered["cluster"] == klaster_target]
    rekomendasi = rekomendasi.sort_values(by=["rating", "ulasan"], ascending=[False, False])

    st.dataframe(rekomendasi[["nama_produk", "harga", "rating", "ulasan", "kategori"]].head(10))

    st.info("💡 Insight: Produk dalam klaster ini punya rating tinggi dan harga terjangkau. Disarankan promosikan dan tingkatkan ulasan pelanggan.")
