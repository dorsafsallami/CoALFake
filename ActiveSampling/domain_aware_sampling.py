import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from ActiveSampling.strategy import Strategy


class DomainAwareSampling(Strategy):
    def __init__(self, annotator_config_name, data_file_path, setting='random', 
                engine='gpt-3.5-turbo', k_range=(3, 10)):
        """
        Maintains original Strategy parameters while adding domain-aware features
        """
        
        super().__init__(
            annotator_config_name=annotator_config_name,
            data_file_path=data_file_path,
            setting=setting,
            engine=engine
        )
        
        
        self.k_min, self.k_max = k_range
        self.clusters = None
        self.cluster_centers = None
        self._preprocess_data()

    def _preprocess_data(self):
        """Convert text data to embeddings after loading"""
        if self.data is not None:
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer('all-mpnet-base-v2')
            self.embeddings = embedder.encode(self.data['text'], show_progress_bar=True)
        else:
            raise ValueError("No data loaded for preprocessing")

    def query(self, args, k: int, model, features):
        """Implementation of domain-aware query strategy"""
        if not hasattr(self, 'embeddings'):
            self._preprocess_data()
            
        k_optimal = self._find_optimal_k()
        self._cluster_data(k_optimal)
        return self._sample_from_clusters(k)

    def _find_optimal_k(self):
        best_score = -1
        best_k = self.k_min
        
        for k in range(self.k_min, self.k_max + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(self.embeddings)
            score = silhouette_score(self.embeddings, labels)
            if score > best_score:
                best_score, best_k = score, k
                
        return best_k

    def _cluster_data(self, k):
        self.kmeans = KMeans(n_clusters=k, random_state=42)
        self.cluster_labels = self.kmeans.fit_predict(self.embeddings)
        self.cluster_centers = self.kmeans.cluster_centers_
        self.clusters = {i: np.where(self.cluster_labels == i)[0] 
                       for i in range(k)}

    def _sample_from_clusters(self, k):
        selected = []
        cluster_counts = {c: len(indices) for c, indices in self.clusters.items()}
        total = sum(cluster_counts.values())
        
        for cluster, indices in self.clusters.items():
            proportion = cluster_counts[cluster] / total
            n_samples = max(1, int(proportion * k))
            distances = np.linalg.norm(
                self.embeddings[indices] - self.cluster_centers[cluster],
                axis=1
            )
            selected.extend(indices[np.argsort(distances)[:n_samples]])
            
        return selected[:k]  