# Movieflix AI - Movie Recommender System 🎬

Movieflix AI adalah sistem rekomendasi film berbasis Machine Learning yang menggunakan dataset **MovieLens 32M**. Aplikasi ini dirancang dengan antarmuka web interaktif ala **Netflix** yang modern dan dinamis, lengkap dengan gambar poster film asli dan deskripsi sinopsis alur cerita yang diunduh secara real-time dari database **The Movie Database (TMDb)**.

🌐 **Demo Live**: [sistem-rekomendasi-film-movielens.streamlit.app](https://sistem-rekomendasi-film-movielens.streamlit.app)

---

## ✨ Fitur Utama

- **Double-Engine Recommendation System**:
  1. **Content-Based Filtering**: Menganalisis kemiripan genre film pilihan Anda dengan film lain menggunakan representasi teks **TF-IDF Vectorizer** dan **Cosine Similarity**.
  2. **Collaborative Filtering**: Menganalisis pola rating jutaan pengguna nyata menggunakan metode **Item-Based Collaborative Filtering** untuk merekomendasikan film sejenis yang disukai bersama oleh penonton lain.
- **Dynamic Scraper (TMDb)**: Menggunakan `requests` dan `BeautifulSoup` untuk melakukan scraping data poster vertikal resmi dan sinopsis alur cerita langsung dari TMDb menggunakan ID tautan dari `links.csv`.
- **Parallel Thread Loading**: Memanfaatkan `ThreadPoolExecutor` di Python untuk mengunduh semua poster dan deskripsi film secara bersamaan (paralel) sehingga hasil rekomendasi termuat secara instan (< 0.8 detik).
- **Out-of-Memory Protection & Caching**: 
  - Memproses file `ratings.csv` raksasa (877 MB, 32 juta rating) menggunakan teknik chunking dan pemfilteran ambang batas aktivitas.
  - Menyimpan matriks korelasi dan statistik rating ke file lokal (`.pkl`) untuk mempercepat pemuatan awal aplikasi.
  - Memanfaatkan caching Streamlit (`@st.cache_data`) agar data film yang pernah dibuka termuat dalam 0 milidetik.
- **Netflix Sleek UI**: Desain bertema gelap pekat (`#141414`), aksen merah khas Netflix (`#E50914`), Cinema Hero Banner di bagian atas, dan visual grid poster film dengan animasi hover zoom-in.
- **Analytics Dashboard**: Grafik statistik interaktif mengenai sebaran tahun rilis film dan 10 genre film paling mendominasi di dalam database.

---

## 📂 Struktur Proyek

```text
├── app.py                # Kode antarmuka (UI) web dashboard Streamlit
├── recommender.py        # Logika rekomendasi, preprocessing, dan TMDb scraper
├── movies.csv            # Data mentah film (ID, judul, genre) - MovieLens 32M
├── ratings.csv           # Data mentah rating (32 juta baris) - MovieLens 32M
├── links.csv             # Data pemetaan ID ke IMDb dan TMDb - MovieLens 32M
├── tags.csv              # Data tag film (tidak digunakan langsung)
├── movie_stats.pkl       # Cache rata-rata rating dan jumlah ulasan per film
├── item_similarity.pkl   # Cache matriks korelasi film untuk Collaborative Filtering
└── README.md             # Dokumentasi proyek (file ini)
```

---

## ⚙️ Persyaratan & Instalasi

Pastikan komputer Anda sudah terinstal **Python 3.10+** dan package manager `pip`.

### 1. Kloning atau Buka Direktori Proyek
Buka terminal/command prompt di direktori proyek ini (`d:\belajar_ml\movie`).

### 2. Instalasi Dependensi
Jalankan perintah berikut untuk menginstal seluruh pustaka Python yang diperlukan:

```bash
pip install pandas numpy scikit-learn requests beautifulsoup4 matplotlib seaborn streamlit
```

---

## 🚀 Cara Menjalankan Aplikasi

Jalankan server aplikasi Streamlit lokal dengan mengetikkan perintah berikut di terminal:

```bash
streamlit run app.py
```

Setelah dijalankan, browser Anda akan otomatis terbuka dan menampilkan Movieflix AI di alamat:
👉 **[http://localhost:8501](http://localhost:8501)**

---

## 🛠️ Detail Teknis & Optimasi Model

1. **Filtering Matrix**:
   Untuk memperkecil ukuran matriks rating dan menghindari kegagalan memory RAM, model Collaborative Filtering memfilter:
   - **Film Populer**: Hanya film yang memiliki minimal **4.000 rating**.
   - **User Aktif**: Hanya pengguna yang telah memberikan minimal **300 rating**.
   Hal ini menghasilkan *utility matrix* berukuran optimal `(26179 user x 1786 film)` yang sangat efisien untuk menghitung cosine similarity antar item.
2. **Fallback Mechanism**:
   Jika film pilihan Anda kurang populer dan belum terdaftar dalam matriks Collaborative Filtering (CF), aplikasi akan otomatis memunculkan peringatan kontekstual dan mengalihkan navigasi penuh ke tab Content-Based Filtering demi kenyamanan pengguna.
