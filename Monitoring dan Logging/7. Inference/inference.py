import requests
import json

url = "http://127.0.0.1:5000/invocations"

# Kolom data
columns = [
    "Time_spent_Alone", "Stage_fear", "Social_event_attendance", "Going_outside",
    "Drained_after_socializing", "Friends_circle_size", "Post_frequency",
    "social_alone_ratio", "friends_and_posts", "drained_by_social"
]

# Data
data_row = [
    1.5997004026219428,    # Time_spent_Alone
    1.0,                   # Stage_fear
    -0.3278600675960191,   # Social_event_attendance
    0.0,                   # Going_outside
    1.0,                   # Drained_after_socializing
    -0.2915532135255693,   # Friends_circle_size
    -0.19074384288667,     # Post_frequency
    -0.6825243279912483,   # social_alone_ratio
    -0.4654072751617367,   # friends_and_posts
    2.1281411632251617     # drained_by_social
]

data = {
    "dataframe_split": {
        "columns": columns,
        "data": [data_row]
    }
}

response = requests.post(url, json=data)
print(response.json())