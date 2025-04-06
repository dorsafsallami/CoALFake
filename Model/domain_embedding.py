import os

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class DomainEmbedding:
    def __init__(self, documents, language_model='all-mpnet-base-v2', k_clusters=5):
        self.documents = documents
        self.model = SentenceTransformer(language_model)  
        self.k_clusters = k_clusters
        self.embeddings = None  

    def embed_documents(self):
        embeddings = self.model.encode(self.documents, convert_to_numpy=True)
        return embeddings

    def perform_clustering(self, embeddings):
        n_samples = embeddings.shape[0]
        max_k = min(n_samples - 1, 10)
        best_k = self.k_clusters
        best_score = -1
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(embeddings)
            try:
                score = silhouette_score(embeddings, kmeans.labels_)
                if score > best_score:
                    best_score = score
                    best_k = k
            except ValueError as e:
                print(f"Skipping k={k} due to error: {e}")
                continue

        self.kmeans = KMeans(n_clusters=best_k, random_state=42)
        self.kmeans.fit(embeddings)
        return best_k

    def calculate_soft_membership(self):
        embeddings = self.embeddings
        centroids = self.kmeans.cluster_centers_

        norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        norm_centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)

        similarities = np.dot(norm_embeddings, norm_centroids.T)

        max_scores = np.max(similarities, axis=1, keepdims=True)
        exp_scores = np.exp(similarities - max_scores)
        softmax_scores = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return softmax_scores

    def get_domain_embeddings(self):
        embeddings = self.embed_documents()
        best_k = self.perform_clustering(embeddings)
        self.embeddings = embeddings
        domain_embeddings = self.calculate_soft_membership()
        return domain_embeddings

if __name__ == '__main__':
    documents = [
        "Climate change is a major threat to biodiversity.",
        "Electric vehicles are becoming more popular.",
        "The new movie received excellent reviews."
    ]
    domain_embedding_generator = DomainEmbedding(documents)
    domain_embeddings = domain_embedding_generator.get_domain_embeddings()
    print("Domain Embeddings (Soft Memberships):\n", domain_embeddings)
    print("Embedding Dimension:", domain_embeddings.shape[1])