from src.data import DataProcessor, TransitDataset
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import joblib
from joblib import Parallel, delayed
import torch

from torch.utils.data import DataLoader

processor = DataProcessor(
    planets=list(Path("ariel-data-challenge-2025/train").glob("*")),
    axis_info=pd.read_parquet("ariel-data-challenge-2025/axis_info.parquet"),
    cache_folder="cached_data"
)
data_path = Path("ariel-data-challenge-2025")

planets_gt = pd.read_csv(data_path / "train.csv")
meta = pd.read_csv(data_path / "train_star_info.csv")

# output_stats = joblib.load("stats.joblib")
output_stats = None

dataset = TransitDataset(processor, planets_gt, meta, output_stats=output_stats)
dataloader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False, persistent_workers=True)

for batch in tqdm(dataloader):
    pass
    # for k, v in batch.items():
    #     if isinstance(v, torch.Tensor):
    #         print(k, v.shape)
    
    # break

for batch in tqdm(dataloader):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape)
    
    break