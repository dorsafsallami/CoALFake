import numpy as np
import torch.nn as nn
from torch import nn

from ActiveSampling.strategy import Strategy


class RandomSampling(Strategy):
    def __init__(self, annotator_config_name, pool_size, setting: str='knn', engine: str='gpt-35-turbo-0301',):
        super().__init__(annotator_config_name, pool_size, setting, engine)

    def query(self, k: int):
        unlabeled_indices = np.where(~self.lab_data_mask)[0]
        sampled_indices = np.random.choice(unlabeled_indices, size=k, replace=False)
        self.lab_data_mask[sampled_indices] = True  
        print("Sampled indices for querying:", sampled_indices)
        return sampled_indices

# Example usage
if __name__ == "__main__":
    strategy = RandomSampling('fake_news_detection', rf'Datasets\training_data.csv')
    sampled_data = strategy.query(50)
    print(sampled_data)