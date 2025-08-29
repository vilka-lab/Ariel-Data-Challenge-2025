from src.data import DataProcessor, TransitDataset
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch

processor = DataProcessor(
    planets=list(Path("ariel-data-challenge-2025/train").glob("*")),
    axis_info=pd.read_parquet("ariel-data-challenge-2025/axis_info.parquet"),
    cache_folder="cached_data"
)

# processor.save("data", plots=True)


# for _ in tqdm(processor):
#     pass


data_path = Path("ariel-data-challenge-2025")
planets_gt = pd.read_csv(data_path / "train.csv")
meta = pd.read_csv(data_path / "train_star_info.csv")
dataset = TransitDataset(processor, planets_gt, meta, output_stats=None)

batch = next(iter(dataset))


for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(k, v.shape)
    