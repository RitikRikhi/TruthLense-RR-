import pandas as pd
from utils.preprocessing import preprocess_text
from sklearn.model_selection import train_test_split
import os

def load_and_prepare_dataset(file_path: str):
    """
    Loads a misinformation dataset from a CSV file.
    Expects the CSV to have at least two columns:
    1. 'text' - The article text or headline
    2. 'label' - 1 for Misleading/Fake, 0 for Credible/Real
    
    Returns:
        train_texts, test_texts, train_labels, test_labels
    """
    
    print(f"Loading dataset from {file_path}...")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}. Please download a dataset (e.g., FakeNewsNet or LIAR) and save it here.")
        
    # 1. Load the dataset using pandas
    df = pd.read_csv(file_path)
    
    # Basic validation to ensure columns exist (can be adjusted based on specific dataset format)
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns.")
        
    print(f"Loaded {len(df)} records. Starting preprocessing...")
    
    # 2. Clean the text
    # We apply the preprocessing logic from our utils to remove URLs, special chars, etc.
    df['clean_text'] = df['text'].apply(preprocess_text)
    
    # Prepare data arrays
    texts = df['clean_text'].tolist()
    labels = df['label'].tolist()
    
    # 3. Split the dataset into train/test
    # 80% for training the model, 20% for testing its accuracy
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    print(f"Dataset split complete: {len(train_texts)} training samples, {len(test_texts)} testing samples.")
    
    return train_texts, test_texts, train_labels, test_labels
    
if __name__ == "__main__":
    # Test script - expects a dummy data file for manual verification
    import os
    os.makedirs("./data", exist_ok=True)
    
    # Create a tiny dummy dataset for testing if none exists
    dummy_csv = "./data/training_data.csv"
    if not os.path.exists(dummy_csv):
        pd.DataFrame({
            "text": ["Scientists discover cure for cancer, click here!", "The Earth is round."],
            "label": [1, 0]
        }).to_csv(dummy_csv, index=False)
        
    load_and_prepare_dataset(dummy_csv)
