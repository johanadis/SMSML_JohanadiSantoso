from prometheus_client import start_http_server, Counter, Gauge, Histogram
import time
import requests
import random
import psutil

# Define feature columns explicitly (excluding 'Personality')
feature_columns = [
    'Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 'Going_outside',
    'Drained_after_socializing', 'Friends_circle_size', 'Post_frequency',
    'social_alone_ratio', 'friends_and_posts', 'drained_by_social'
]

# Define binary features 
binary_features = ['Stage_fear', 'Drained_after_socializing']

# Initialize Prometheus metrics
request_count = Counter('model_requests_total', 'Total requests to model')
prediction_latency = Histogram('model_prediction_latency_seconds', 'Prediction latency in seconds')
error_count = Counter('model_errors_total', 'Total errors in predictions')
success_rate = Gauge('model_success_rate', 'Success rate of predictions')
cpu_usage = Gauge('model_cpu_usage_percent', 'CPU usage percentage')
memory_usage = Gauge('model_memory_usage_percent', 'Memory usage percentage')
predicted_class = Counter('model_predicted_class_total', 'Total predictions for each class', ['class'])

# Function to send request to the model server
def send_request_to_model():
    # Generate random input data
    input_data = []
    for col in feature_columns:
        if col in binary_features:
            # Binary features: 0.0 or 1.0
            value = random.choice([0.0, 1.0])
        else:
            # Continuous features: uniform distribution between -2 and 2
            value = random.uniform(-2, 2)
        input_data.append(value)
    
    # Prepare data in the expected JSON format
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
            # Assuming prediction is a float representing the class label (0.0 or 1.0)
            class_label = str(int(prediction))
            predicted_class.labels(class=class_label).inc()
        else:
            error_count.inc()
            success_rate.set(0.0)
    except Exception:
        error_count.inc()
        success_rate.set(0.0)

# Function to collect system metrics
def collect_system_metrics():
    cpu_usage.set(psutil.cpu_percent())
    memory_usage.set(psutil.virtual_memory().percent)

# Main loop to start the metrics server and send requests
if __name__ == '__main__':
    start_http_server(8000)
    print("Prometheus metrics server running on port 8000")
    while True:
        send_request_to_model()
        collect_system_metrics()
        time.sleep(1)