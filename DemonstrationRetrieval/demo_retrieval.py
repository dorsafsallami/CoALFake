import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from embedding import get_cosine_similarity, get_embeddings

print(torch.cuda.is_available())

def load_data(file_path):
    return pd.read_csv(file_path)

def get_tensor_from_embeddings(df, column_name):
    embeddings = torch.tensor(df[column_name].tolist())
    return embeddings.to(args.device) 

def convert_to_native_types(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def knn_demo_retrieval(args):
    dir_path = os.path.join(os.getcwd(), rf'Datasets')  
    data = {'demo': load_data(os.path.join(dir_path, 'demo_sample.csv')),
            'train': load_data(os.path.join(dir_path, 'training_data.csv'))}
    
    test_sets = ['gossip_test', 'politifact_test', 'coaid_test']
    for test_set in test_sets:
        data[test_set] = load_data(os.path.join(dir_path, f'{test_set}.csv'))
        

    for key in data:
        embeddings = get_embeddings(args, data[key]['text'])
        data[key]['embeddings'] = [emb.tolist() for emb in embeddings]  

    for test_set in test_sets:
        train_embeddings = get_tensor_from_embeddings(data['train'], 'embeddings')
        demo_embeddings = get_tensor_from_embeddings(data['demo'], 'embeddings')

        sim = get_cosine_similarity(train_embeddings, demo_embeddings)
        scores, indices = sim.topk(k=args.topk, dim=-1)
        results = {}

        num_iterations = min(len(data[test_set]), scores.size(0))
        for i in range(num_iterations):
            key_id = int(data[test_set].iloc[i]['id']) 
            context = []
            for score, idx in zip(scores[i], indices[i]):
                if idx.item() < len(data['demo']):
                    val_id = data['demo'].iloc[idx.item()]['id']
                    score = score.item()
                    context.append({'id': val_id, 'score': score})
                else:
                    print(f"Index {idx.item()} is out of bounds for 'demo' DataFrame size {len(data['demo'])}")

                if len(context) == args.topk:
                    break

            results[str(key_id)] = context
        dst_path = os.path.join(dir_path, f'{test_set}-knn-demo.json')
        with open(dst_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, default=convert_to_native_types, ensure_ascii=False, indent=2)


def random_demo_retrieval(args):
    random.seed(args.seed)
    dir_path = os.path.join(os.getcwd(), rf'Datasets')

    data = {'demo': load_data(os.path.join(dir_path, 'demo_sample.csv')),
            'train': load_data(os.path.join(dir_path, 'training_data.csv'))}
    
    test_sets = ['gossip_test', 'politifact_test', 'coaid_test']
    for test_set in test_sets:
        data[test_set] = load_data(os.path.join(dir_path, f'{test_set}.csv'))
    
    demo_ids = data['demo']['id'].tolist()
    demo_ids = [int(id) for id in demo_ids]

    for test_set in test_sets:
        results = {}
        for i in range(len(data[test_set])):
            key_id = int(data[test_set].iloc[i]['id'])
            context = random.sample(demo_ids, k=args.topk)
            context = [{'id': int(id)} for id in context]
            results[str(key_id)] = context
        dst_path = os.path.join(dir_path, f'{test_set}-random-demo.json')
        with open(dst_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='fakenews', type=str)
    parser.add_argument('--topk', default=5, type=int)
    parser.add_argument('--embedding_model_name_or_path',
                        default='paraphrase-multilingual-mpnet-base-v2', type=str)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--seed', default=3407, type=int)
    parser.add_argument('--method', default='knn', type=str)
    args = parser.parse_args()

    if args.method == 'random':
        random_demo_retrieval(args)
    elif args.method == 'knn':
        knn_demo_retrieval(args)
    else:
        raise ValueError('Unknown retrieval method.')
