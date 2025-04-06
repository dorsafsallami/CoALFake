import pandas as pd
import pytest
from annotator import Annotator
from label_verification import find_label_issues
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
    
    demo_texts = [
        "Chocolate helps lose weight",
        "Water freezes at 0°C", 
        "Aliens built the pyramids",  
        "Humans have 23 pairs of chromosomes",  
        "Vaccines cause autism",  
        "The Great Wall is visible from space",  
        "Plants absorb carbon dioxide", 
        "5G causes COVID-19", 
        "The heart has four chambers",  
        "Humans use only 10% of their brain"  
    ]
    
    annotator = Annotator(engine='gpt-3.5-turbo')
    demo_labels = [annotator.online_annotate({"text": t}) for t in demo_texts]
    demo_labels[-1] = "Real" 
    
    data = pd.DataFrame({"text": demo_texts, "llm_label": demo_labels})
    
    print("\nGenerated Labels:")
    print(data)
    
    clf = LogisticRegression(max_iter=1000)
    issues = find_label_issues(clf, data, percentage=0.1)
    
    print("\nDetected Issues:")
    print(issues)
    
    