
from pathlib import Path
import json, sys
import pandas as pd
import xgboost as xgb
import yaml
ROOT=Path(__file__).resolve().parent
with open(ROOT/"configs/config.yaml") as f: yaml.safe_load(f)
for p in ["data/customers.csv","data/orders.csv","data/returns.csv","data/products.csv","data/splits.csv",
          "models/artifacts/model_a.json","models/artifacts/model_a_meta.json",
          "models/artifacts/model_b.json","models/artifacts/model_b_meta.json",
          "eval/results/phase1_model_a_test_metrics.json","eval/results/phase2_adaptive_eval_metrics.json",
          "eval/results/phase3_model_b_eval_metrics.json"]:
    assert (ROOT/p).exists(), p
for m in ["model_a","model_b"]:
    b=xgb.Booster(); b.load_model(str(ROOT/f"models/artifacts/{m}.json"))
print("OK: config, data, metrics, and both frozen model artifacts load.")
print("Valid customer:", pd.read_csv(ROOT/"data/customers.csv").iloc[3]["customer_id"])
