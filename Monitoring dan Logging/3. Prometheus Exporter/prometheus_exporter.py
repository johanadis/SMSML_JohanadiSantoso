from prometheus_client import start_http_server, Counter, Gauge, Histogram
import time
import requests
import random
import psutil

# Daftar kolom fitur yang digunakan dalam model, tidak termasuk 'Personality' (label target)
feature_columns = [
    'Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 'Going_outside',
    'Drained_after_socializing', 'Friends_circle_size', 'Post_frequency',
    'social_alone_ratio', 'friends_and_posts', 'drained_by_social'
]

# Fitur kategorikal (biner) yang hanya memiliki nilai 0.0 atau 1.0
categorical_cols = ['Stage_fear', 'Drained_after_socializing']

# Inisialisasi metrik Prometheus untuk memantau kinerja model
request_count = Counter('model_requests_total', 'Total permintaan ke model')  # Menghitung total permintaan
prediction_latency = Histogram('model_prediction_latency_seconds', 'Latensi prediksi dalam detik')  # Mengukur latensi prediksi
error_count = Counter('model_errors_total', 'Total kesalahan dalam prediksi')  # Menghitung kesalahan prediksi
success_rate = Gauge('model_success_rate', 'Tingkat keberhasilan prediksi')  # Mengukur tingkat keberhasilan (1.0 = sukses, 0.0 = gagal)
cpu_usage = Gauge('model_cpu_usage_percent', 'Persentase penggunaan CPU')  # Mengukur penggunaan CPU
memory_usage = Gauge('model_memory_usage_percent', 'Persentase penggunaan memori')  # Mengukur penggunaan memori
predicted_class = Counter('model_predicted_class_total', 'Total prediksi untuk setiap kelas', ['class'])  # Menghitung prediksi per kelas

# Fungsi untuk mengirim permintaan prediksi ke server model
def send_request_to_model():
    # Hasilkan data input acak untuk pengujian
    input_data = []
    for col in feature_columns:
        if col in categorical_cols:
            value = random.choice([0.0, 1.0])  # Nilai acak untuk fitur kategorikal
        else:
            value = random.uniform(-2, 2)  # Nilai acak untuk fitur numerikal
        input_data.append(value)
    
    # Siapkan data dalam format JSON untuk server model
    data = {
        "dataframe_split": {
            "columns": feature_columns,
            "data": [input_data]
        }
    }
    url = "http://127.0.0.1:5000/invocations"  # URL server model
    start_time = time.time()  # Catat waktu mulai untuk mengukur latensi
    try:
        response = requests.post(url, json=data)  # Kirim permintaan POST
        latency = time.time() - start_time  # Hitung latensi
        prediction_latency.observe(latency)  # Catat latensi
        request_count.inc()  # Tambah hitungan permintaan
        if response.status_code == 200:
            success_rate.set(1.0)  # Set tingkat keberhasilan ke 1.0
            prediction = response.json().get('predictions', [0])[0]  # Ambil prediksi
            class_label = str(int(prediction))  # Konversi prediksi ke label kelas
            predicted_class.labels(**{'class': class_label}).inc()  # Tambah hitungan prediksi kelas
        else:
            error_count.inc()  # Tambah hitungan kesalahan
            success_rate.set(0.0)  # Set tingkat keberhasilan ke 0.0
    except Exception:
        error_count.inc()  # Tambah hitungan kesalahan jika terjadi exception
        success_rate.set(0.0)  # Set tingkat keberhasilan ke 0.0

# Fungsi untuk mengumpulkan metrik penggunaan sistem
def collect_system_metrics():
    cpu_usage.set(psutil.cpu_percent())  # Set penggunaan CPU
    memory_usage.set(psutil.virtual_memory().percent)  # Set penggunaan memori

# Loop utama untuk memulai server metrik dan mengirim permintaan secara berkala
if __name__ == '__main__':
    start_http_server(8000)  # Mulai server HTTP Prometheus di port 8000
    print("Server metrik Prometheus berjalan di port 8000")
    while True:
        send_request_to_model()  # Kirim permintaan ke model
        collect_system_metrics()  # Kumpulkan metrik sistem
        time.sleep(1)  # Tunggu 1 detik sebelum iterasi berikutnya