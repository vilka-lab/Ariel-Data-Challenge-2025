from pathlib import Path
import joblib

import pandas as pd
import joblib
import lightning as L
import torch
from sklearn.model_selection import KFold
from src.generator import GeneratedDataProcessor
from src.data import TransitDataset, DataProcessor, CombinedProcessor



class TransitDataModule(L.LightningDataModule):
    def __init__(
            self, 
            data_path: str, 
            batch_size: int = 16,
            num_workers: int = 0,
            test_size: float = 0.2,
            random_state: int = 42,
            full_train: bool = False,
            fold: int = 0
            ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_path = Path(data_path)
        self.test_size = test_size
        self.random_state = random_state
        self.full_train = full_train
        self.fold = fold

        self.planets_stop_list = [
            "926530491",
            "905997089",
            "1124834224",
            "806204363",
            "158006264",
            "158006264",
            "561423413",
            "1843015807",
            "1338107575"
        ]


    def setup(self, stage: str | None = None, load_stats: bool = False) -> None:
        planets = sorted(list((self.data_path / "train").glob("*")))

        # if not self.full_train:
        #     train_planets, test_planets = train_test_split(
        #         planets, test_size=self.test_size, random_state=self.random_state
        #     )
        # else:
        #     train_planets, test_planets = planets, planets

        kfold = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        train_indices, test_indices = list(kfold.split(planets))[self.fold]
        train_planets = [planets[i] for i in train_indices]
        test_planets = [planets[i] for i in test_indices]

        print("before", len(train_planets), len(test_planets))

        train_planets = [p for p in train_planets if p.name not in self.planets_stop_list]
        test_planets = [p for p in test_planets if p.name not in self.planets_stop_list]
        
        print("after", len(train_planets), len(test_planets))

        planets_gt = pd.read_csv(self.data_path / "train.csv")
        meta = pd.read_csv(self.data_path / "train_star_info.csv")
        axis_info = pd.read_parquet(self.data_path / "axis_info.parquet")

        train_processor = DataProcessor(train_planets, axis_info=axis_info, cache_folder="cached_data")
        # df = pd.read_csv("generated_data.csv")
        # wavelengths = pd.read_csv("ariel-data-challenge-2025/wavelengths.csv")
        # train_processor2 = GeneratedDataProcessor(df, cache_folder="cached_generated_data", wavelengths=wavelengths)

        # train_processor = CombinedProcessor([train_processor1, train_processor2])
        test_processor = DataProcessor(test_planets, axis_info=axis_info, cache_folder="cached_data")

        stats_path = Path(f"stats_{self.fold}.joblib")
        if load_stats and stats_path.exists():
            output_stats = joblib.load(stats_path)
        else:
            output_stats = None

        # generated_gt = df[planets_gt.columns]
        # planets_gt = pd.concat([planets_gt, generated_gt])

        # generated_meta = df[meta.columns]
        # meta = pd.concat([meta, generated_meta])

        self.train_dataset = TransitDataset(train_processor, planets_gt, meta, output_stats=output_stats)
        joblib.dump(self.train_dataset.get_stats(), stats_path)

        self.val_dataset = TransitDataset(test_processor, planets_gt, meta, output_stats=self.train_dataset.get_stats())


    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )