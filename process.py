from src.data import DataProcessor
import pandas as pd
from pathlib import Path
from tqdm import tqdm

processor = DataProcessor(
    planets=list(Path("ariel-data-challenge-2025/train").glob("*")),
    axis_info=pd.read_parquet("ariel-data-challenge-2025/axis_info.parquet"),
    cache_folder="data"
)

# processor.save("data", plots=True)


for _ in tqdm(processor):
    pass