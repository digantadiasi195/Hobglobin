#generate_chat_responses.py
import requests

questions = open("sample_questions.txt").readlines()
with open("chat_response.txt", "w") as f:
    for q in questions:
        q = q.strip()
        if q:
            response = requests.post("http://localhost:8000/chat", json={"query": q})
            f.write(f"Q: {q}\nA: {response.json()['response']}\n\n")