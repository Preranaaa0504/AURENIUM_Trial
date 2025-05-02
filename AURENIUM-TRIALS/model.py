import requests

api_key = "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"

response = requests.get(
    "https://openrouter.ai/api/v1/models",
    headers={
        "Authorization": f"Bearer {api_key}"
    }
)

if response.status_code == 200:
    models = response.json()
    for model in models['data']:
        print(f"ID: {model['id']}, Name: {model['name']}")
else:
    print(f"Error {response.status_code}: {response.text}")
