import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

size = 1000

gt_df = pd.read_csv("ariel-data-challenge-2025/train.csv")
meta_df = pd.read_csv("ariel-data-challenge-2025/train_star_info.csv")
df = meta_df.merge(gt_df, on="planet_id")


fields = ["P", "sma", "i", "e", "Rs", "Ms", "Ts", "Mp", "wl_1"]

stats = {}
for field in fields:
    stats[field] = {}
    stats[field]["mean"] = df[field].mean()
    stats[field]["std"] = df[field].std()

for i in range(2, 284):
    coef = df[f"wl_{i}"] / df[f"wl_{i - 1}"]
    stats[f"wl_{i}_coef"] = {}
    stats[f"wl_{i}_coef"]["mean"] = coef.mean()
    stats[f"wl_{i}_coef"]["std"] = coef.std()
    fields.append(f"wl_{i}_coef")

stats["D"] = {
    "mean": 25.0,
    "std": 1.0
}
fields.append("D")

stats["e"] = {
    "mean": 0,
    "std": 0.05
}

methods = ["quadratic", "squareroot", "logarithmic", "exponential"]


generated_data = np.stack([np.random.normal(stats[f]["mean"], stats[f]["std"], size=size) for f in fields], axis=1)
generated_df = pd.DataFrame(generated_data, columns=fields)

generated_df["wl_1"] = generated_df["wl_1"].abs()
generated_df["e"] = generated_df["e"].abs()

generated_df["t0"] = np.random.uniform(0.6, 0.95, size=size) * generated_df["P"]

for i in range(2, 284):
    generated_df[f"wl_{i}"] = (generated_df[f"wl_{i - 1}"] * generated_df[f"wl_{i}_coef"]).values
    generated_df = generated_df.drop(f"wl_{i}_coef", axis=1).copy()

generated_df["planet_id"] = np.arange(len(generated_df))
# generated_df["method"] = np.random.choice(methods, size=size)
generated_df["method"] = "quadratic"

def make_coef(r: pd.Series) -> pd.Series:
    # r["ld_coeffs"] = np.random.random(size=2).tolist()
    r["ld_coeffs"] = [0.1, 0.3]
    return r

generated_df = generated_df.apply(make_coef, axis=1)

cols = [f"wl_{i}" for i in range(1, 284)]

def filter(r: pd.Series) -> pd.Series:
    vals = r[cols].values
    vals = savgol_filter(vals, window_length=7, polyorder=3)
    r[cols] = vals
    return r

generated_df = generated_df.apply(filter, axis=1)

generated_df.to_csv("generated_data.csv", index=False)