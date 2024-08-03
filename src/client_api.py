import requests

base_url = "http://localhost:8000"

# def preprocess(text):
#     response = requests.post(f"{base_url}/preprocess/", json={"text": text})
#     return response.json()

def sentiment(text):
    response = requests.post(f"{base_url}/sentiment/", json={"text": text})
    return response.json()

# def generate(text):
#     response = requests.post(f"{base_url}/generate/", json={"text": text})
#     return response.json()

if __name__ == "__main__":
    text = "J'adore le nouveau design du produit ! Il est incroyable et fonctionne parfaitement."
    # print("Preprocessing:", preprocess(text))
    print("Sentiment:", sentiment(text))
    # print("Generate:", generate(text))
