from src.generator import GeneratedDataProcessor
import pandas as pd
from tqdm import tqdm

df = pd.read_csv("generated_data.csv")
wavelengths = pd.read_csv("ariel-data-challenge-2025/wavelengths.csv")
processor = GeneratedDataProcessor(df, cache_folder="cached_generated_data", wavelengths=wavelengths)

for _ in tqdm(processor):
    pass