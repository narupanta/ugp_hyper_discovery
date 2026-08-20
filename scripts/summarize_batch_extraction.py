import os
import json
import argparse
import pandas as pd
import numpy as np
from glob import glob

def summarize(batch_dir, output_csv):
    # Find all metrics files in the batch dir
    pattern = os.path.join(batch_dir, "**", "extraction_metrics.json")
    metrics_files = glob(pattern, recursive=True)
    
    data = []
    
    for metrics_file in metrics_files:
        with open(metrics_file, "r") as f:
            try:
                metrics = json.load(f)
            except json.JSONDecodeError:
                continue
                
        # Format lists into strings for CSV
        for k, v in metrics.items():
            if isinstance(v, list):
                metrics[k] = str(v)
                
        data.append(metrics)
        
    if not data:
        print(f"No extraction_metrics.json found in '{batch_dir}'.")
        return
        
    df = pd.DataFrame(data)
    
    # Reorder columns to ensure seed is first
    cols = df.columns.tolist()
    if 'seed' in cols:
        cols.insert(0, cols.pop(cols.index('seed')))
        df = df[cols]
        
    # Sort by seed
    if 'seed' in df.columns:
        df = df.sort_values(by='seed').reset_index(drop=True)
        
    # Calculate Mean and Std
    # Only calculate for numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    mean_row = numeric_df.mean().to_dict()
    std_row = numeric_df.std().to_dict()
    
    mean_row['seed'] = 'Mean'
    std_row['seed'] = 'Std'
    
    # Append as new rows
    df = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    
    df.to_csv(output_csv, index=False)
    print(f"Summary saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize batch extraction metrics")
    parser.add_argument("batch_dir", type=str, help="Batch directory containing seed folders")
    args = parser.parse_args()
    
    output_csv = os.path.join(args.batch_dir, "batch_summary.csv")
    summarize(args.batch_dir, output_csv)
