# Impor modul yang diperlukan untuk Prometheus, pengukuran waktu, permintaan HTTP, pembuatan data acak, dan pengukuran penggunaan sistem
from prometheus_client import start_http_server, Counter, Gauge, Histogram
import time
import requests
import random
import psutil

# Definisikan kolom fitur yang digunakan dalam model machine learning
# Kolom ini tidak termasuk 'Personality' yang merupakan label target
feature_columns = [
    'Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 'Going_outside',
    'Drained_after_socializing', 'Friends_circle_size', 'Post_frequency',
    'social_alone_ratio', 'friends_and_posts', 'drained_by_social'
]

# Definisikan fitur yang bersifat kategorikal (biner) berdasarkan pengamatan data
# Fitur ini hanya memiliki dua kemungkinan nilai: 0.0 atau 1.0
categorical_cols = ['Stage_fear', 'Drained_after_socializing']

# Inisialisasi metrik Prometheus untuk memantau kinerja model
# request_count: Menghitung total permintaan yang dikirim ke model
request_count = Counter('model_requests_total', 'Total permintaan ke model')
# prediction_latency: Mengukur distribusi latensi prediksi dalam detik
prediction_latency = Histogram('model_prediction_latency_seconds', 'Latensi prediksi dalam detik')
# error_count: Menghitung total kesalahan yang terjadi saat prediksi
error_count = Counter('model_errors_total', 'Total kesalahan dalam prediksi')
# success_rate: Mengukur tingkat keberhasilan prediksi (1.0 jika sukses, 0.0 jika gagal)
success_rate = Gauge('model_success_rate', 'Tingkat keberhasilan prediksi')
# cpu_usage: Mengukur persentase penggunaan CPU sistem
cpu_usage = Gauge('model_cpu_usage_percent', 'Persentase penggunaan CPU')
# memory_usage: Mengukur persentase penggunaan memori sistem
memory_usage = Gauge('model_memory_usage_percent', 'Persentase penggunaan memori')
# predicted_class: Menghitung total prediksi untuk setiap kelas (0 atau 1)
predicted_class = Counter('model_predicted_class_total', 'Total prediksi untuk setiap kelas', ['class'])

# Fungsi untuk mengirim permintaan prediksi ke server model
def send_request_to_model():
    # Hasilkan data input acak untuk pengujian
    # Data ini mensimulasikan input yang akan diprediksi oleh model
    input_data = []
    for col in feature_columns:
        if col in categorical_cols:
            # Untuk fitur kategorikal, pilih nilai acak antara 0.0 atau 1.0
            value = random.choice([0.0, 1.0])
        else:
            # Untuk fitur numerikal, pilih nilai acak dari distribusi uniform antara -2 dan 2
            value = random.uniform(-2, 2)
        input_data.append(value)
    
    # Siapkan data dalam format JSON yang diharapkan oleh server model
    # Format ini sesuai dengan apa yang diharapkan oleh endpoint '/invocations'
    data = {
        "dataframe_split": {
            "columns": feature_columns,
            "data": [input_data]
        }
    }
    url = "http://127.0.0.1:5000/invocations"
    start_time = time.time()  # Catat waktu mulai untuk mengukur latensi
    try:
        # Kirim permintaan POST ke server model
        response = requests.post(url, json=data)
        latency = time.time() - start_time  # Hitung latensi
        prediction_latency.observe(latency)  # Catat latensi ke metrik
        request_count.inc()  # Tambah hitungan permintaan
        if response.status_code == 200:
            # Jika permintaan sukses, set success_rate ke 1.0
            success_rate.set(1.0)
            # Ambil prediksi dari respons JSON
            prediction = response.json().get('predictions', [0])[0]
            # Prediksi adalah float yang mewakili label kelas (0.0 = Introvert atau 1.0 = Extrovert)
            class_label = str(int(prediction))
            # Tambah hitungan untuk kelas yang diprediksi
            predicted_class.labels(class=class_label).inc()
        else:
            # Jika permintaan gagal, tambah hitungan kesalahan
            error_count.inc()
            success_rate.set(0.0)
    except Exception:
        # Jika terjadi exception, tambah hitungan kesalahan
        error_count.inc()
        success_rate.set(0.0)

# Fungsi untuk mengumpulkan metrik penggunaan sistem (CPU dan memori)
def collect_system_metrics():
    # Set metrik CPU usage dengan persentase penggunaan CPU saat ini
    cpu_usage.set(psutil.cpu_percent())
    # Set metrik memory usage dengan persentase penggunaan memori saat ini
    memory_usage.set(psutil.virtual_memory().percent)

# Loop utama untuk memulai server metrik dan mengirim permintaan secara berkala
if __name__ == '__main__':
    # Mulai server HTTP Prometheus di port 8000
    start_http_server(8000)
    print("Server metrik Prometheus berjalan di port 8000")
    while True:
        # Kirim permintaan ke model
        send_request_to_model()
        # Kumpulkan metrik sistem
        collect_system_metrics()
        # Tunggu 1 detik sebelum iterasi berikutnya
        time.sleep(1)