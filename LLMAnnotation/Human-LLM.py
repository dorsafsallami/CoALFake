
import csv
import os
import time

import pandas as pd
import pytest
from annotator import Annotator
from func_timeout import FunctionTimedOut, func_timeout
from label_verification import find_label_issues
from sklearn.linear_model import LogisticRegression


def clean_temp_csv(file_path):
    """Ensures all rows in temp CSV have correct formatting before reading."""
    cleaned_rows = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 3:  
                cleaned_rows.append(row)

    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(cleaned_rows)

def process_csv_with_llm(input_path, output_path, temp_output_path="temp_results.csv"):
    df = pd.read_csv(input_path)
    total_rows = len(df)

    if os.path.exists(temp_output_path):
        clean_temp_csv(temp_output_path) 
        df_saved = pd.read_csv(temp_output_path)
        print(f"Resuming from {temp_output_path}, {len(df_saved)} rows already processed.")
    else:
        df_saved = pd.DataFrame(columns=['text', 'llm_label', 'llm_cost'])  

    processed_texts = set(df_saved['text']) if not df_saved.empty else set()

    print("Starting LLM annotation...")
    annotator = Annotator(engine='gpt-3.5-turbo')

    with open(temp_output_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        for row_number, (_, row) in enumerate(df.iterrows(), start=1):
            text = row['text']

            if text in processed_texts:
                continue

            print(f"Processing row {row_number} of {total_rows}...")

            for attempt in range(3):
                try:
                    result, cost = func_timeout(
                        120, 
                        lambda: annotator.online_annotate({"text": text}, return_cost=True)
                    )
                    break
                except FunctionTimedOut:
                    print(f"Timeout on attempt {attempt+1} for row {row_number}...")
                    if attempt == 2:
                        result, cost = "TIMEOUT_ERROR", 0.0
                    time.sleep(2 ** attempt)
                except Exception as e:
                    print(f"Error processing row {row_number}: {text[:50]}... | {str(e)}")
                    result, cost = "API_ERROR", 0.0
                    break

            writer.writerow([text, result, cost])
            f.flush()

    clean_temp_csv(temp_output_path)
    df_annotated = pd.read_csv(temp_output_path)
    df = df.merge(df_annotated, on='text', how='left')

    print("Starting Confident Learning...")
    clf = LogisticRegression(max_iter=1000)
    percentages = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

    for p in percentages:
        print(f"Processing {int(p*100)}% correction...")
        issues = find_label_issues(clf, df, percentage=p)
        
        col_name = f'corrected_{int(p*100)}%'
        df[col_name] = df.apply(
            lambda row: row['label'] if row.name in issues.index else row['llm_label'],
            axis=1
        )
        
        replaced_col = f'replaced_{int(p*100)}%'
        df[replaced_col] = df.index.isin(issues.index)

    df.to_csv(output_path, index=False)
    os.remove(temp_output_path)
    print(f"\nProcessed file saved to {output_path}")

    return df


