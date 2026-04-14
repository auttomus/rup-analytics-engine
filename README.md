# RUP Spatial Analytics Engine

RUP Spatial Analytics Engine adalah dasbor analitik interaktif berbasis Streamlit yang dirancang untuk mengeksplorasi, mengklasifikasikan, dan memvisualisasikan data Rencana Umum Pengadaan (RUP) Pemerintah.

Sistem ini mengimplementasikan pendekatan *Machine Learning* (K-Means & HDBSCAN) yang dipadukan dengan pemrosesan bahasa alami (NLP) untuk secara otomatis mengelompokkan paket pengadaan berdasarkan kesamaan semantik teks (Nama Paket) serta metadata kategorial (Matriks Dimensi/Tag).

## Fitur Utama

* **Dynamic Clustering:** Mendukung algoritma K-Means dan HDBSCAN dengan pembobotan dinamis antara fitur teks (TF-IDF/Embeddings) dan fitur kategorial (Tag Matrix).
* **Dimensionality Reduction:** Visualisasi ruang kluster 2D interaktif menggunakan metode t-SNE, UMAP, atau PCA.
* **Calibration Tools:** Dilengkapi dengan analisis matematis komprehensif, termasuk *Elbow Method*, *Silhouette Score*, dan *HDBSCAN Min Size Advisor*, guna menentukan parameter kluster yang optimal secara empiris.
* **Comprehensive Visualization:** Fasilitas eksplorasi data secara mendalam melalui *Scatter Plots*, *Bar Distributions*, *Treemaps*, dan hierarki *Sunburst*.
* **Modular Architecture:** Struktur kode terpisah secara logis (`components`, `modules`, `config`, `data`) untuk menjamin skalabilitas dan kemudahan pemeliharaan di masa mendatang.

---

## Prasyarat dan Instalasi

Pastikan sistem Anda telah terinstal **Python 3.10** atau versi yang lebih baru. Sangat disarankan untuk menggunakan *virtual environment* guna mengisolasi dependensi program.

1. **Kloning Repositori**

   ```bash
   git clone https://github.com/auttomus/rup-analytics-engine.git
   cd rup-analytics-engine
   ```
2. **Pembuatan & Aktivasi Virtual Environment**
   **Bash**

   ```
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Instalasi Dependensi**
   **Bash**

   ```
   pip install -r requirements.txt
   ```
4. **Menjalankan Aplikasi**
   **Bash**

   ```
   streamlit run app.py
   ```

---

## Panduan Penggunaan dan Praktik Analisis Terbaik

Untuk memperoleh hasil klasterisasi yang presisi dan relevan secara operasional, terapkan alur kerja analitik berikut:

### 1. Persiapan & Input Data

Gunakan panel *Sidebar* untuk menentukan sumber data. Anda dapat memilih file CSV yang telah tersedia di direktori `data/` atau mengunggah ( *upload* ) file CSV RUP baru. File CSV RUP dapat diunduh dari [Data INAPROC](https://data.inaproc.id/rup). Pastikan struktur kolom pada berkas sesuai dengan standar yang didefinisikan pada `config/settings.py`.

### 2. Kalibrasi Parameter (Fase Kritis)

Sebelum menganalisis hasil akhir, buka tab  **Kalibrasi** . Ini adalah langkah fundamental untuk menghindari asumsi acak.

* **Penggunaan K-Means:** Perhatikan metrik *Elbow* dan  *Silhouette* . Tentukan K pada titik di mana kurva *Inertia* melandai tajam dan skor *Silhouette* mencapai puncaknya. Masukkan nilai tersebut pada pengaturan **Jumlah Kluster (K)** di  *sidebar* .
* **Penggunaan HDBSCAN:** Tekan tombol  **Jalankan Sweep** . Engine akan melakukan iterasi untuk mencari nilai `min_cluster_size` optimal yang menghasilkan rasio *Noise* terendah dengan skor *Silhouette* tertinggi.

### 3. Penyesuaian Pembobotan (Teks vs. Tag)

Pada bagian **Fitur Clustering** di panel kontrol, sesuaikan rasio pada indikator  **Bobot Tag vs Teks** .

* **Teks Dominan (Rasio rendah):** Direkomendasikan apabila deskripsi pada "Nama Paket" sangat spesifik dan bervariasi.
* **Tag Dominan (Rasio tinggi):** Direkomendasikan apabila "Nama Paket" bersifat generik (misalnya: "Belanja Modal", "Honorarium"), sehingga mesin perlu mengandalkan parameter struktural seperti "Cara Pengadaan" atau "Metode Pengadaan".

### 4. Eksplorasi Visual

Setelah parameter ditetapkan, lakukan inspeksi data melalui tab visualisasi yang tersedia:

* **Statistik Kluster & Treemap:** Identifikasi kluster dengan serapan anggaran dan konsentrasi volume tertinggi.
* **Sunburst:** Petakan hierarki pengadaan dari Cara Pengadaan berlanjut ke Jenis dan Metode Pengadaan.
* **Scatter Plot:** Gunakan dimensi aktif untuk menerapkan pewarnaan pada titik data, guna mendeteksi pola sebaran atau anomali distribusi paket.

### 5. Ekstraksi Data

Akses tab  **Data** , terapkan filter pada kluster spesifik yang menjadi fokus analisis, lalu gunakan fungsi **Download CSV** untuk mengekspor dataset beserta label kluster yang telah digenerasi untuk keperluan pelaporan lebih lanjut.

---

## Pelaporan Bug dan Permintaan Fitur

 Apabila Anda menemukan anomali pada sistem atau memiliki usulan peningkatan (seperti integrasi model NLP baru atau penyempurnaan dasbor metrik), silakan ajukan laporan melalui [Issues](https://github.com/auttomus/rup-analytics-engine/issues).

 Untuk mempermudah proses investigasi terkait  *bug* , mohon sertakan informasi berikut:

* Sistem operasi dan versi Python yang digunakan.
* Langkah-langkah rinci untuk mereproduksi *bug* tersebut.
* *Stack trace* atau log pesan galat ( *error* ) terkait.

---

## Panduan Kontribusi

Pengembangan kolaboratif sangat didorong. Jika Anda berminat berkontribusi pada repositori ini, ikuti standar protokol berikut:

1. Lakukan *Fork* pada repositori utama.
2. Buat *branch* fitur baru (`git checkout -b feature/NamaFitur`).
3. Lakukan *commit* pada modifikasi Anda (`git commit -m 'Implementasi NamaFitur'`).
4. Unggah ke repositori *fork* Anda (`git push origin feature/NamaFitur`).
5. Ajukan **Pull Request** ke repositori utama.

Harap pastikan kode yang diajukan mematuhi struktur modular arsitektur program dan menghindari penggunaan *magic numbers* (gunakan `config/settings.py` untuk penyesuaian konstanta).

---

Dikembangkan oleh auttomus
