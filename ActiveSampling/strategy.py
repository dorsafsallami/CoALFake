import os
import time
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import ujson as json
from func_timeout.exceptions import FunctionTimedOut
from openai.error import RateLimitError
from torch import nn

from LLMAnnotation import Annotator

RETRY = 3

class Strategy(ABC):
    def __init__(self, annotator_config_name, data_file_path, setting='random', engine='gpt-3.5-turbo'):
        
        if not os.path.exists(data_file_path):
            raise FileNotFoundError(f"Data file not found: {data_file_path}")
        
        self.data = pd.read_csv(data_file_path)
        print(f"Successfully loaded {len(self.data)} samples")

        try:
            self.data = pd.read_csv(data_file_path)
            self.lab_data_mask = np.zeros(len(self.data), dtype=bool)  
            print(f"Data loaded successfully, data shape: {self.data.shape}")
        except Exception as e:
            print(f"Failed to load data from {data_file_path}: {e}")
            self.data = None
        if self.data is not None:
            pass
        else:
            raise ValueError("Data could not be loaded and is required for further processing.")


    def __len__(self):
        return len(self.lab_data_mask)

    def _get_labeled_indices(self):
        return np.where(self.lab_data_mask)[0]
    
    def _get_pool_indices(self):
        return np.where(~self.lab_data_mask)[0]

    def get_labeled_data(self, features):
        label_key = 'label'  
        results = [features[i] for i in self._get_labeled_indices() if features[i][label_key] is not None]
        return results

    @abstractmethod
    def query(self, args, k: int, model: nn.Module, features):
        pass

    def init_labeled_data(self, n_sample: int=None):
        if n_sample is None:
            raise ValueError('Please specify initial sample ratio/size.')
        assert n_sample <= len(self)

        indices = np.arange(len(self))
        np.random.shuffle(indices)
        indices = indices[: n_sample]
        self.lab_data_mask[:] = False
        return indices
    
    def update(self, indices, features):
        self.lab_data_mask[indices] = True
        return self.annotate(features)

    def annotate(self, features):
        results = {}
        for i in self._get_labeled_indices():
            label_key = 'label'
            if features[i][label_key] is None:
                demo = None
                if self.setting != 'zero':
                    demo = [self.demo_file.get(pointer['id']) for pointer in sorted(self.demo_file.keys())] 
                result = None
                for j in range(RETRY):
                    try:
                        result = self.annotator.online_annotate(features[i], demo)
                        break
                    except FunctionTimedOut:
                        print('Timeout. Retrying...')
                    except RateLimitError:
                        print('Rate limit. Sleep for 60 seconds...')
                        time.sleep(60)
                results[features[i]['id']] = result
        print('Annotate {} new records.'.format(len(results)))
        return results
