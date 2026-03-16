import re
import string

def preprocess_text(text: str) -> str:
    """
    Cleans raw text input to prepare it for the model pipeline.
    
    Why preprocessing is important:
    1. Removes noise (URLs, special characters) that the model doesn't understand.
    2. Standardizes text (lowercase) so "Warning" and "warning" are treated the same.
    3. Prevents the model from learning irrelevant patterns.
    """
    
    # 1. Remove URLs using regular expressions
    # This prevents the model from attempting to interpret website links
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # 2. Convert to lowercase
    # Standardization step for consistency
    text = text.lower()
    
    # 3. Remove punctuation and special characters
    # Punctuation rarely helps in misinformation detection and adds noise
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 4. Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

if __name__ == "__main__":
    # Example usage for testing
    sample = "BREAKING! Scientists discovered miracle cure for cancer at https://fake-news.com... Check it out!!!"
    print(f"Original: {sample}")
    print(f"Cleaned: {preprocess_text(sample)}")
