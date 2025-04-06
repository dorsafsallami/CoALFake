import copy
from typing import Any

import pandas as pd
from cleanlab.classification import CleanLearning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder


def find_label_issues(clf: Any, data: pd.DataFrame, percentage: float = 0.1) -> pd.DataFrame:
    """
    Find label issues using CleanLearning, prioritizing lowest-quality disagreements.

    Parameters:
    - clf: The base classifier.
    - data: A DataFrame containing 'text' and 'llm_label' columns.
    - percentage: The percentage of samples to select for reannotation (e.g., 0.1 for 10%).

    Returns:
    - A DataFrame containing the selected samples for reannotation.
    """
    texts = data["text"].values
    labels = data["llm_label"].values
    
    base_clf = make_pipeline(
        TfidfVectorizer(),
        copy.deepcopy(clf)
    )
    
    min_samples = min(pd.Series(labels).value_counts())
    cv_n_folds = min(3, min_samples)
    cl = CleanLearning(base_clf, cv_n_folds=max(2, cv_n_folds))
    
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    
    label_issues = cl.find_label_issues(X=texts, labels=encoded_labels)
    
    lowest_quality_indices = label_issues["label_quality"].argsort().to_numpy()
    
    num_samples = max(1, int(len(lowest_quality_indices) * percentage))
    
    results_df = pd.DataFrame({
        "text": texts,
        "llm_label": labels,
        "suggested_label": encoder.inverse_transform(label_issues["predicted_label"]),
        "label_quality": label_issues["label_quality"]
    })
    
    return results_df.iloc[lowest_quality_indices].head(num_samples)
