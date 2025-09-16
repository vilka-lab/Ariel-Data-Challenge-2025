import random
import numpy as np


class AbstractAugmentation:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img: np.ndarray):
        if random.random() < self.p:
            for i in range(img.shape[0]):
                img[i] = self.augment(img[i])
        
        return img
    

class VerticalFlip(AbstractAugmentation):
    def augment(self, img: np.ndarray) -> np.ndarray:
        return img[::-1, ...].copy()
    

class GaussianNoise(AbstractAugmentation):
    def __init__(self, p: float = 0.5, std: float = 0.1) -> None:
        super().__init__(p)
        self.std = std

    def augment(self, img: np.ndarray) -> np.ndarray:
        return img + np.random.normal(0, self.std, img.shape).astype(img.dtype)
    

class EmptyVerticalLines(AbstractAugmentation):
    def __init__(self, p: float = 0.5, num_lines: int = 10) -> None:
        super().__init__(p)
        self.num_lines = num_lines

    def augment(self, img: np.ndarray) -> np.ndarray:
        _, w = img.shape

        num_lines = random.randint(1, self.num_lines)
        values = np.random.choice([0, img.max()], size=num_lines)

        lines = random.sample(range(w), num_lines)
        img[:, lines] = values[None, :]
        return img
    
class EmptyHorizontalLines(AbstractAugmentation):
    def __init__(self, p: float = 0.5, num_lines: int = 10) -> None:
        super().__init__(p)
        self.num_lines = num_lines

    def augment(self, img: np.ndarray) -> np.ndarray:
        h, _ = img.shape

        num_lines = random.randint(1, self.num_lines)
        values = np.random.choice([0, img.max()], size=num_lines)

        lines = random.sample(range(h), num_lines)
        img[lines, :] = values[:, None]
        return img
    

class AugmentationList:
    def __init__(self, augmentations: list[AbstractAugmentation]) -> None:
        self.augmentations = augmentations

    def __call__(self, img: np.ndarray):
        for augmentation in self.augmentations:
            img = augmentation(img)
        
        return img
    
    def __len__(self):
        return len(self.augmentations)


def get_augmentations(stage: str) -> AugmentationList:
    if stage == "train":
        return AugmentationList([
            VerticalFlip(p=0.5),
            GaussianNoise(p=0.3, std=0.005),
            EmptyHorizontalLines(p=0.2, num_lines=5),
            EmptyVerticalLines(p=0.2, num_lines=5),
        ])
    else:
        return AugmentationList([])