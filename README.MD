# Proyek Akhir: Submission Membangun Sistem Machine Learning untuk Klasifikasi Kepribadian | SMSML_JohanadiSantoso

## Ikhtisar Proyek
Proyek ini bertujuan untuk membangun sistem machine learning yang mengklasifikasikan kepribadian individu sebagai ekstrover atau introver berdasarkan data perilaku. Proyek ini memenuhi panduan submission Dicoding Indonesia dengan fokus pada empat kriteria utama: eksperimen dataset, pelatihan model, workflow CI, serta monitoring dan alerting.

## Dataset
Proyek ini menggunakan dataset [Extrovert vs. Introvert Behavior Data](https://www.kaggle.com/datasets/rakeshkapilavai/extrovert-vs-introvert-behavior-data) dari Kaggle. Dataset ini berisi data perilaku yang relevan untuk klasifikasi kepribadian ekstrover vs. introver.

## CI/CD Pipeline
Proyek ini dilengkapi dengan pipeline Continuous Integration dan Continuous Deployment (CI/CD) menggunakan GitHub Actions dan MLflow. Pipeline ini mengotomatisasi langkah-langkah berikut:

1. **Checkout Kode**: Mengambil kode terbaru dari repositori menggunakan `actions/checkout@v3`.
2. **Setup Python**: Mengkonfigurasi lingkungan Python 3.12.11 dengan `actions/setup-python@v4`.
3. **Instal Dependensi**: Menginstal paket Python yang diperlukan, seperti MLflow, scikit-learn, dan lainnya.
4. **Atur MLflow Tracking URI**: Mengkonfigurasi URI pelacakan MLflow untuk melacak eksperimen.
5. **Jalankan Proyek MLflow**: Melatih model dengan `mlflow run .` dan mencatat hasil pelatihan ke `mlflow_output.log`.
6. **Ekstrak Run IDs**: Mengambil ID run dari pelatihan untuk digunakan selanjutnya.
7. **Unduh Artefak**: Mengunduh model yang telah dilatih dan artefak lainnya ke `mlruns/0/<run_id>`.
8. **Unggah ke Git LFS**: Menyimpan file besar menggunakan Git Large File Storage.
9. **Bangun Image Docker**: Membuat image Docker menggunakan `MLProject/Dockerfile`.
10. **Login ke Docker Hub**: Melakukan autentikasi ke Docker Hub dengan `docker/login-action@v2`.
11. **Tag dan Push Image**: Memberi tag dan mengunggah image Docker ke Docker Hub.
12. **Lokal: Tarik Image Docker**: Instruksi untuk menarik image Docker secara lokal.
13. **Lokal: Jalankan Kontainer**: Perintah untuk menjalankan kontainer Docker dengan model.
14. **Lokal: Uji API**: Cara untuk menguji API model yang telah dideploy menggunakan `curl`.

