import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Judul aplikasi
st.title("Prediksi Customer Clustering")
st.write("Masukkan file yang berisi daftar pembelian suatu retail.")

# Model
model_clus3 = joblib.load('model_clus3.pkl')

# Mengunggah file
uploaded_file = st.file_uploader("Unggah file Excel Anda", type=["xlsx", "xls"])

if uploaded_file is not None:
    try:
        # Membaca file Excel
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        
        # Membersihkan header kolom
        df.columns = df.columns.str.strip()
        
        # Menambahkan kolom Amount
        df['Amount'] = df['Quantity'] * df['UnitPrice']

        # Menampilkan isi file
        st.write("### Data Transaksi:")
        st.dataframe(df)

        # Pastikan kolom StockCode menjadi string dan tidak ada nilai NaN
        df['StockCode'] = df['StockCode'].astype(str)
        df['StockCode'] = df['StockCode'].fillna('Unknown')

        # Analisis RFM
        required_columns = ['CustomerID', 'InvoiceDate', 'InvoiceNo', 'Amount']
        if all(col in df.columns for col in required_columns):

            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

            # Hitung Recency
            recency = df.groupby('CustomerID')['InvoiceDate'].max().reset_index()
            recency['recency'] = (recency['InvoiceDate'].max() - recency['InvoiceDate']).dt.days + 1

            # Hitung Frequency
            frequency = df.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
            frequency.columns = ['CustomerID', 'frequency']

            # Hitung Monetary
            monetary = df.groupby('CustomerID')['Amount'].sum().reset_index()
            monetary.columns = ['CustomerID', 'monetary']

            # DataFrame RFM
            rfm = recency[['CustomerID', 'recency']].merge(frequency, on='CustomerID').merge(monetary, on='CustomerID')
            st.write("### Data RFM:")
            st.dataframe(rfm)

            # assigning the numbers to RFM values. The better the RFM value higher the number
            # note that this process is reverse for recency score as lower the value the better it is
            rfm['recency_score'] = pd.cut(rfm['recency'], bins=[0, 18, 51, 143, 264, 375], labels=[5, 4, 3, 2, 1])
            rfm['recency_score'] = rfm['recency_score'].astype('int')
            rfm['frequency_score'] = pd.cut(rfm['frequency'], bins=[0, 1, 2, 5, 9, 210], labels=[1, 2, 3, 4, 5])
            rfm['frequency_score'] = rfm['frequency_score'].astype('int')
            rfm['monetary_score'] = pd.cut(rfm['monetary'], bins=[-1, 306, 667, 1650, 3614, 290000], labels=[1, 2, 3, 4, 5])
            rfm['monetary_score'] = rfm['monetary_score'].astype('int')

            # RFM score
            def score_rfm(x):
                return (x['recency_score']) + (x['frequency_score']) + (x['monetary_score'])

            rfm['score'] = rfm.apply(score_rfm, axis=1)
            st.write("### Data RFM dengan Skor:")
            st.dataframe(rfm)

            # Menggunakan model KMeans untuk memprediksi cluster
            df_rfm_for_clustering = rfm[['recency', 'frequency', 'monetary']]
            df_rfm_for_clustering = StandardScaler().fit_transform(df_rfm_for_clustering)
            rfm['cluster'] = model_clus3.predict(df_rfm_for_clustering)

            # Menampilkan hasil prediksi cluster
            st.write("### Hasil Prediksi Cluster untuk Masing-Masing Pelanggan:")
            st.dataframe(rfm[['CustomerID', 'recency', 'frequency', 'monetary', 'score', 'cluster']])

            # Deskripsi Singkat Setiap Cluster
            cluster_descriptions = {
                0: "Cluster 0: Pelanggan dengan tingkat pembelian tinggi tetapi jarang bertransaksi. Mereka memiliki recency yang cukup tinggi, namun sering membeli barang dengan nilai transaksi yang tinggi.",
                1: "Cluster 1: Pelanggan yang sangat aktif, sering bertransaksi dan menghasilkan pembelian yang signifikan. Mereka memiliki frekuensi yang tinggi dan pembelian yang besar.",
                2: "Cluster 2: Pelanggan dengan nilai transaksi tinggi, tetapi jarang bertransaksi. Mereka berpotensi menjadi pelanggan VIP dengan nilai monetari tinggi meskipun jarang berinteraksi."
            }

            # Menampilkan Deskripsi Cluster dengan Tabel Pelanggan
            for cluster, description in cluster_descriptions.items():
                st.subheader(f"Cluster {cluster}")
                st.write(description)
                
                # Tampilkan tabel pelanggan yang masuk ke dalam cluster tersebut
                cluster_df = rfm[rfm['cluster'] == cluster]
                st.write(f"### Tabel Pelanggan pada Cluster {cluster}:")
                st.dataframe(cluster_df[['CustomerID', 'recency', 'frequency', 'monetary', 'score']])

        else:
            missing_cols = [col for col in required_columns if col not in df.columns]
            st.error(f"Kolom yang diperlukan tidak ditemukan: {', '.join(missing_cols)}")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file: {e}")
else:
    st.info("Silakan unggah file Excel untuk melihat datanya.")
