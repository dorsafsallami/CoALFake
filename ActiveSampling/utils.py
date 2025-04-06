import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from Model.model import FAKE_NEWS_DETECTOR


def fake_news_predict(args, features, model: FAKE_NEWS_DETECTOR):
    dataloader = DataLoader(features, args.test_batch_size, shuffle=False)
    model.eval()
    pred_logits = []
    for batch in tqdm(dataloader, desc='Evaluating on pool data'):
        inputs = {'input_ids': batch['input_ids'].to(args.device),
                  'attention_mask': batch['attention_mask'].to(args.device)}

        with torch.no_grad():
            outputs = model.forward(**inputs)
            logits = outputs 
            attention_mask = batch['attention_mask'].to(args.device)

            for i in range(logits.size(0)):
                pred = logits[i][attention_mask[i] == 1].cpu()
                pred_logits.append(pred)
    
    return pred_logits

def get_fake_news_embeddings(args, features, model: nn.Module, normalize: bool=True):
 
    model.eval()
    embeddings = []
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False)

    for batch in tqdm(dataloader, desc='Computing embeddings'):
        inputs = {'input_ids': batch['input_ids'].to(args.device),
                  'attention_mask': batch['attention_mask'].to(args.device)}

        with torch.no_grad():
            outputs = model(**inputs)
            attention_mask = inputs['attention_mask']
            outputs = (outputs * attention_mask.unsqueeze(-1)).sum(dim=1)
            outputs = outputs / attention_mask.sum(dim=1, keepdim=True)

            if normalize:
                outputs = F.normalize(outputs, p=2, dim=-1)

            embeddings.append(outputs)

    embeddings = torch.cat(embeddings, dim=0)
    return embeddings


