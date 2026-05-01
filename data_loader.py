import pandas as pd
from datasets import load_dataset
import os

def main():
    """
    Downloads the nbertagnolli/counsel-chat dataset from HuggingFace,
    cleans it, formats each row, and saves it to data/knowledge_base.csv.
    """
    print("Loading dataset 'nbertagnolli/counsel-chat'...")
    dataset = load_dataset("nbertagnolli/counsel-chat", split="train")
    
    # Convert to pandas DataFrame for easier manipulation
    df = dataset.to_pandas()
    
    print("Cleaning and formatting data...")
    # Keep only relevant columns if they exist.
    if 'questionText' in df.columns and 'answerText' in df.columns:
        df['formatted_text'] = "Q: " + df['questionText'].astype(str) + " A: " + df['answerText'].astype(str)
    else:
        # Fallback if column names differ
        col1 = df.columns[0]
        col2 = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        df['formatted_text'] = "Q: " + df[col1].astype(str) + " A: " + df[col2].astype(str)

    # Drop NA values
    df = df.dropna(subset=['formatted_text'])
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV
    output_path = 'data/knowledge_base.csv'
    df[['formatted_text']].to_csv(output_path, index=False)
    print(f"Data saved successfully to {output_path}. Total records: {len(df)}")

if __name__ == "__main__":
    main()
