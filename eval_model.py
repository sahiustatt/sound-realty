# eval_model.py
import os, json, pickle
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

DATA_DIR = "data"
MODEL_DIR = "model"

# 1) Load data with correct dtypes
houses = pd.read_csv(os.path.join(DATA_DIR, "kc_house_data.csv"),
                     dtype={"zipcode": str})
demo   = pd.read_csv(os.path.join(DATA_DIR, "zipcode_demographics.csv"),
                     dtype={"zipcode": str})

# 2) Merge demographics on zipcode
df = houses.merge(demo, on="zipcode", how="left")

# 3) Target and drop zipcode (model was not trained on raw zipcode)
y = df["price"]
df = df.drop(columns=["price", "zipcode"])

# 4) Load trained model + exact feature order used in training
with open(os.path.join(MODEL_DIR, "model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(MODEL_DIR, "model_features.json"), "r") as f:
    model_features = json.load(f)

# 5) Safety checks: ensure we have the exact columns the model expects
have = set(df.columns)
need = set(model_features)

missing = [c for c in model_features if c not in have]
extra   = [c for c in df.columns if c not in need]
if missing:
    raise RuntimeError(f"Missing columns required by model: {missing}")
# (It's fine to have 'extra' columns; we simply won't use them.)

# 6) Select columns in the exact order used during training
X = df[model_features]

# 7) Evaluate
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
yhat_tr = model.predict(Xtr)
yhat_te = model.predict(Xte)

print("Train R2:", round(r2_score(ytr, yhat_tr), 3))
print("Test  R2:", round(r2_score(yte, yhat_te), 3))
print("Test  MAE:", round(mean_absolute_error(yte, yhat_te), 0))

