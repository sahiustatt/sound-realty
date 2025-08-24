# test_api.py
import pandas as pd, requests

API = "http://127.0.0.1:8000"
df = pd.read_csv("data/future_unseen_examples.csv", dtype={"zipcode": str})

minimal_keys = ["bedrooms","bathrooms","sqft_living","sqft_lot",
                "floors","sqft_above","sqft_basement","zipcode"]

for i, row in df.head(5).iterrows():
    full_payload = row.to_dict()
    r1 = requests.post(f"{API}/predict", json=full_payload).json()

    mini_payload = {k: full_payload[k] for k in minimal_keys}
    r2 = requests.post(f"{API}/predict_minimal", json=mini_payload).json()

    print(f"[{i}] zip={full_payload['zipcode']}  full=${r1['predicted_price']:.0f}  "
          f"minimal=${r2['predicted_price']:.0f}")

