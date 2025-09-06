import pandas as pd
import requests
from io import BytesIO
from dotenv import load_dotenv
import os

load_dotenv()
dbtwin_api_key = os.getenv("DBTWIN_API_KEY")
num_rows = 10000

file_url = "https://github.com/LNshuti/kora/blob/backend_kg/synthetic_data.csv"
request = requests.get(file_url)
file_obj = BytesIO(request.content)

file_obj.seek(0)
files = {"file": ("synthetic_data.csv", file_obj)}
headers = {"rows": num_rows, "algo": "flagship", "api-key": dbtwin_api_key}

url = "https://api.dbtwin.com"
print(requests.get(url + "/health"))

resp = requests.post(url + "/generate", headers=headers, files=files)
if resp.status_code == 200:
    with open("synthetic_data.csv", "wb") as f:
        f.write(resp.content)
    df_synth = pd.read_csv("synthetic_data.csv")
    print(df_synth.head())
    print(resp.headers['distribution-similarity-error'])
    print(resp.headers['association-similarity'])

else:
    print(resp.json())

