import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Judul aplikasi
st.title("Prediksi Customer Clustering")
st.write("Masukkan file yang berisi daftar pembelian customer.")

# Model

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
            # note that this process is reverse for R score as lower the value the better it is
            
            rfm['recency_score'] = pd.cut(rfm['recency'], bins=[0,18,51,143,264,375], labels=[5,4,3,2,1])
            rfm['recency_score'] = rfm['recency_score'].astype('int')
            rfm['frequency_score'] = pd.cut(rfm['frequency'], bins=[0,1,2,5,9,210], labels=[1,2,3,4,5])
            rfm['frequency_score'] = rfm['frequency_score'].astype('int')
            rfm['monetary_score'] = pd.cut(rfm['monetary'], bins=[-1,306,667,1650,3614,290000], labels=[1,2,3,4,5])
            rfm['monetary_score'] = rfm['monetary_score'].astype('int')

            # RFM score
            def score_rfm(x) : return (x['recency_score']) + (x['frequency_score']) + (x['monetary_score']) / 3
            rfm['score'] = rfm.apply(score_rfm,axis=1 )
            rfm.head()

        else:
            missing_cols = [col for col in required_columns if col not in df.columns]
            st.error(f"Kolom yang diperlukan tidak ditemukan: {', '.join(missing_cols)}")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file: {e}")
else:
    st.info("Silakan unggah file Excel untuk melihat datanya.")



# 
# with st.form('prediction'):
# 
#   sepal_length = st.slider("Sepal Length (cm)", min_value=4.0, max_value=8.0, step=0.1, value=5.0)
# 
#   sepal_width = st.slider("Sepal Width (cm)", min_value=2.0, max_value=4.5, step=0.1, value=3.5)
# 
#   petal_length = st.slider("Petal Length (cm)", min_value=1.0, max_value=7.0, step=0.1, value=1.4)
# 
#   petal_width = st.slider("Petal Width (cm)", min_value=0.1, max_value=2.5, step=0.1, value=0.2)
# 

# 
#   submit_button = st.form_submit_button(label='Predict')
# 

# 
# if submit_button:
# 
#   new_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
# 
#   predicted_cluster = model.predict(new_data)
# 
#   st.subheader("Hasil Prediksi:")
#   st.write(f"Data baru masuk ke cluster: *{predicted_cluster[0]}*")