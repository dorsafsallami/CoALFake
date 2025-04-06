import torch
from kmeans_pytorch import kmeans
from torch import nn

from ActiveSampling.strategy import Strategy
from ActiveSampling.utils import get_fake_news_embeddings


class KMeansSampling(Strategy):
    def __init__(self, annotator_config_name, pool_size, setting: str = 'random', engine: str='gpt-3.5-turbo'):
        super().__init__(annotator_config_name, pool_size, setting, engine)

    def query(self, args, k: int, model: nn.Module, features):
        pool_indices = self._get_pool_indices()
        pool_features = [features[i] for i in pool_indices]
        embeddings = get_fake_news_embeddings(args, pool_features, model)
        
        ids, centers = kmeans(X=embeddings, num_clusters=k, device=args.device)
        
        device = embeddings.device
        centers = centers.to(device)
        
        dist = torch.cdist(centers, embeddings)
        min_distances, lab_indices = torch.min(dist, dim=1)
        lab_indices = [pool_indices[i] for i in lab_indices]

        return lab_indices
