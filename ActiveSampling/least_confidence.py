import torch
from torch import nn

from ActiveSampling.strategy import Strategy
from ActiveSampling.utils import fake_news_predict


class LeastConfidence(Strategy):
    def __init__(self, annotator_config_name, pool_size, setting: str='random', engine: str='gpt-3.5-turbo',
                 reduction: str='mean'):
        super().__init__(annotator_config_name, pool_size, setting, engine)
        assert reduction in ['mean', 'sum', 'min']
        self.reduction = reduction

    def query(self, args, k: int, model: nn.Module, features):
        pool_indices = self._get_pool_indices()
        pool_features = [features[i] for i in pool_indices]
        
        logits = fake_news_predict(args, pool_features, model)
        prob = torch.softmax(logits, dim=-1)
        
        confidence = 1 - torch.max(prob, dim=-1)[0]

        if self.reduction == 'mean':
            uncertainties = confidence.mean()
        elif self.reduction == 'sum':
            uncertainties = confidence.sum()
        elif self.reduction == 'min':
            uncertainties = confidence.max()
        else:
            raise ValueError(f'Unsupported reduction method: {self.reduction}')
        
        _, topk_indices = torch.topk(uncertainties, k=k)
        lab_indices = [pool_indices[i] for i in topk_indices]

        return lab_indices
