import requests
import json
try:
    base_url = "http://localhost:9997/v1"
    response = requests.get(f"{base_url}/models")
    models = response.json()
    with open("models_list.json", "w") as f:
        json.dump(models, f)
    print("Done")
except Exception as e:
    with open("error.log", "w") as f:
        f.write(str(e))
