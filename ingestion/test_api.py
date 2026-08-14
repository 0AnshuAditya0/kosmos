import requests

url = "https://datasets-server.huggingface.co/rows"
params = {
    "dataset": "ai4bharat/MSMARCO-XI",
    "config": "default",
    "split": "train",
    "offset": 0,
    "length": 3,
}

response = requests.get(url, params=params, timeout=30)
print("Status code:", response.status_code)
print(response.json())