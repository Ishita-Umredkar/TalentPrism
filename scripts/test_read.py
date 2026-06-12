import requests

response = requests.post(
    "http://127.0.0.1:8080/v1/chat/completions",
    json={
        "messages": [
            {
                "role": "user",
                "content": "What is 2+2?"
            }
        ],
        "temperature": 0,
        "max_tokens": 50
    }
)

print(response.status_code)
print(response.text)