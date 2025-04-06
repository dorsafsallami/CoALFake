import argparse
import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

from ActiveSampling.domain_aware_sampling import DomainAwareSampling
from LLMAnnotation import Annotator
from LLMAnnotation.label_verification import find_label_issues
from Model.domain_embedding import DomainEmbedding
from Model.model import FAKE_NEWS_DETECTOR

DATA_PATH = r'Datasets\training_data.csv'
ANNOTATOR_CONFIG_PATH = r'LLMAnnotation\configs\fake_news_detection.json'

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Training data not found at: {DATA_PATH}")
    
    data = pd.read_csv(DATA_PATH)
    data = data[['text', 'label']].drop_duplicates()
    print(f"Loaded dataset with {len(data)} samples")

    
    # Split Data into Train and Test
    train_data, test_data = train_test_split(data, test_size=0.2, stratify=data['label'], random_state=42)
    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    
    # Compute Domain-Aware Embeddings
    print("Computing domain-aware embeddings for training data...")
    domain_embedder_train = DomainEmbedding(train_data['text'].tolist(), language_model='all-mpnet-base-v2')
    train_text_embeddings = domain_embedder_train.embed_documents()
    train_domain_embeddings = domain_embedder_train.get_domain_embeddings()
    train_data['embedding'] = list(train_text_embeddings)
    train_data['domain_embedding'] = list(train_domain_embeddings)
    
    print("Computing domain-aware embeddings for test data...")
    domain_embedder_test = DomainEmbedding(test_data['text'].tolist(), language_model='all-mpnet-base-v2')
    test_text_embeddings = domain_embedder_test.embed_documents()
    test_domain_embeddings = domain_embedder_test.get_domain_embeddings()
    test_data['embedding'] = list(test_text_embeddings)
    test_data['domain_embedding'] = list(test_domain_embeddings)

    # Create Initial Labeled and Unlabeled Pools
    initial_labeled = train_data.sample(frac=0.1, random_state=42).reset_index(drop=True)
    unlabeled_pool = train_data.drop(initial_labeled.index).reset_index(drop=True)
    
    # Initialize the Detector Using the Initial Labeled Pool
    X_train_initial = np.stack(initial_labeled['embedding'])
    yd_train_initial = np.stack(initial_labeled['domain_embedding']).astype(np.float32)

    input_dim = X_train_initial.shape[1]
    domain_emb_dim = yd_train_initial.shape[1]

    lambda1, lambda2, lambda3 = 1, 1, 0.5
    lambda4, lambda5 = 0.1, 0.1
    latent_dim = 512
    epochs = 300
    batch_size = 128

    detector = FAKE_NEWS_DETECTOR(
        input_d=input_dim,
        domain_emb_d=domain_emb_dim,
        latent_d=latent_dim,
        lambda1=lambda1,
        lambda2=lambda2,
        lambda3=lambda3,
        lambda4=lambda4,
        lambda5=lambda5
    )
    print("Initial training on the labeled data...")
    detector.train(
        X_train=X_train_initial,
        y_train=initial_labeled['label'].values,
        yd_train=yd_train_initial,
        epochs=epochs,
        batch_size=batch_size
    )

    # Prepare the LLM Annotator
    annotator = Annotator(
        engine='gpt-3.5-turbo',
        config_path=ANNOTATOR_CONFIG_PATH
    )
   
    # Active Learning Loop
    metrics_history = []
    for cycle in range(10):
        print(f"\n=== ACTIVE LEARNING CYCLE {cycle+1}/10 ===")
        
        # Domain-aware sample selection
        sampler = DomainAwareSampling(
            annotator_config_name="fake-news",
            data_file_path=DATA_PATH,
            setting='active',
            engine='gpt-3.5-turbo',
            k_range=(10, 10)
        )
        
        selected_indices = sampler.query(
           args=None,
           k=100,
           model=detector.generator,
           features=unlabeled_pool
        )

        valid_indices = [idx for idx in selected_indices if idx in unlabeled_pool.index]
        if len(valid_indices) != len(selected_indices):
            print(f"Filtered {len(selected_indices)-len(valid_indices)} invalid indices")
        
        if not valid_indices:
            print("No valid indices selected, skipping cycle")
            continue

        batch = unlabeled_pool.loc[valid_indices]

        
        # LLM Annotation of Selected Samples
        print("Annotating samples with LLM...")
        llm_labels = []
        costs = []
        for idx, row in batch.iterrows():
            success = False
            for attempt in range(3):
                try:
                    result, cost = annotator.online_annotate(
                        {"text": row['text']}, 
                        return_cost=True
                    )
                    llm_labels.append(result)
                    costs.append(cost)
                    success = True
                    break
                except Exception as e:
                    print(f"Annotation error (attempt {attempt+1}): {str(e)}")
                    time.sleep(2 ** attempt)
            if not success:
                llm_labels.append(-1) 
                costs.append(0)
        
        batch['llm_label'] = llm_labels
        batch['annotation_cost'] = costs
        
        # Label Verification
        print("Verifying labels...")
        batch['verified_label'] = batch['label']
        mismatch_mask = batch['llm_label'] != batch['verified_label']
        correction_rate = mismatch_mask.mean()
        print(f"Label correction rate: {correction_rate:.2%}")
        
        valid_batch = batch[batch['llm_label'] != -1]
        unlabeled_pool = unlabeled_pool.drop(valid_indices).reset_index(drop=True)
        initial_labeled = pd.concat([initial_labeled, valid_batch], ignore_index=True)
        
        # Update Domain Embeddings for the Updated Labeled Pool
        print("Updating domain representations for labeled data...")
        domain_embedder = DomainEmbedding(initial_labeled['text'].tolist(), language_model='all-mpnet-base-v2')
        updated_domain_embeddings = domain_embedder.get_domain_embeddings()
        initial_labeled['domain_embedding'] = list(updated_domain_embeddings)
        
        # Fine-Tune the Detector with the Updated Labeled Pool
        X_train_current = np.stack(initial_labeled['embedding'])
        y_train_current = initial_labeled['verified_label'].values
        yd_train_current = np.stack(initial_labeled['domain_embedding']).astype(np.float32)
        print("Fine-tuning detector model...")
        detector.train(
            X_train=X_train_current,
            y_train=y_train_current,
            yd_train=yd_train_current,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Evaluate the Model on the Test Set
        X_test = np.stack(test_data['embedding'])
        y_test = test_data['label'].values
        yd_test = np.stack(test_data['domain_embedding']).astype(np.float32)
        metrics = detector.evaluate(X_test, y_test, yd_test)
        metrics_history.append(metrics)
        
        print(f"\nCycle {cycle+1} Metrics:")
        print(f"- Accuracy: {metrics['accuracy']:.4f}")
        print(f"- F1 Score: {metrics['f1']:.4f}")
    
   
    # Save Final Results
    print("\nSaving results...")
    initial_labeled.to_csv("final_labeled_dataset.csv", index=False)
    pd.DataFrame(metrics_history).to_csv("training_metrics.csv", index=False)
    print("Process completed successfully!")

if __name__ == "__main__":
    main()