from prometheus_client import start_http_server, Counter, Gauge, Histogram
import time
import requests
import random
import psutil

feature_columns = [
    'Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 'Going_outside',
    'Drained_after_socializing', 'Friends_circle_size', 'Post_frequency',
    'social_alone_ratio', 'friends_and_posts', 'drained_by_social'
]

categorical_cols = ['Stage_fear', 'Drained_after_socializing']

request_count = Counter('model_requests_total', 'Total permintaan ke model')
prediction_latency = Histogram('model_prediction_latency_seconds', 'Latensi prediksi dalam detik')
error_count = Counter('model_errors_total', 'Total kesalahan dalam prediksi')
success_rate = Gauge('model_success_rate', 'Tingkat keberhasilan prediksi')
cpu_usage = Gauge('model_cpu_usage_percent', 'Persentase penggunaan CPU')
memory_usage = Gauge('model_memory_usage_percent', 'Persentase penggunaan memori')
predicted_class = Counter('model_predicted_class_total', 'Total prediksi untuk setiap kelas', ['class'])

def send_request_to_model():
    input_data = []
    for col in feature_columns:
        if col in categorical_cols:
            value = random.choice([0.0, 1.0])
        else:
            value = random.uniform(-2, 2)
        input_data.append(value)
    
    data = {
        "dataframe_split": {
            "columns": feature_columns,
            "data": [input_data]
        }
    }
    url = "http://127.0.0.1:5000/invocations"
    start_time = time.time()
    try:
        response = requests.post(url, json=data)
        latency = time.time() - start_time
        prediction_latency.observe(latency)
        request_count.inc()
        if response.status_code == 200:
            success_rate.set(1.0)
            prediction = response.json().get('predictions', [0])[0]
            class_label = str(int(prediction))
            predicted_class.labels(**{'class': class_label}).inc()
        else:
            error_count.inc()
            success_rate.set(0.0)
    except Exception:
        error_count.inc()
        success_rate.set(0.0)

def collect_system_metrics():
    cpu_usage.set(psutil.cpu_percent())
    memory_usage.set(psutil.virtual_memory().percent)

if __name__ == '__main__':
    start_http_server(8000)
    print("Server metrik Prometheus berjalan di port 8000")
    while True:
        send_request_to_model()
        collect_system_metrics()
        time.sleep(1)