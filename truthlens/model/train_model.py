import os
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from model.dataset_loader import load_and_prepare_dataset
import threading

# We load environment variables here or in main, assuming we'll handle this in an orchestrator.
# For simplicity in this script, we'll accept paths as arguments.

def encode_dataset(texts, labels, tokenizer):
    """
    Tokenizes the text using the DistilBERT tokenizer so the model can understand it.
    """
    # Tokenization converts words to numbers (tokens)
    # Padding ensures all sequences have the same length
    # Truncation ensures no sequence is longer than the model's max limit
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
    
    # We create a simple PyTorch dataset class format required by the HuggingFace Trainer
    class TruthLensDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)
            
    return TruthLensDataset(encodings, labels)

def start_training(data_path: str, model_save_path: str):
    """
    1. Loads dataset
    2. Tokenizes dataset
    3. Fine-tunes the DistilBERT model
    4. Saves the trained model locally
    """
    
    # Ensure the save directory exists
    os.makedirs(model_save_path, exist_ok=True)
    
    # Load and split dataset using Pandas
    try:
        train_texts, test_texts, train_labels, test_labels = load_and_prepare_dataset(data_path)
    except Exception as e:
        print(f"Dataset loading failed: {e}")
        return

    # Load pre-trained DistilBERT tokenizer
    # We use DistilBERT because it's lighter and faster than BERT, ideal for hackathons
    print("Loading tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    
    # Tokenize the data
    print("Tokenizing data...")
    train_dataset = encode_dataset(train_texts, train_labels, tokenizer)
    test_dataset = encode_dataset(test_texts, test_labels, tokenizer)
    
    # Load the pre-trained DistilBERT model for sequence classification
    # num_labels=2 because we are classifying as "Credible" (0) or "Misleading" (1)
    print("Loading base model...")
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',          # Output directory
        num_train_epochs=3,              # Train for 3 epochs (iterations over the dataset)
        per_device_train_batch_size=16,  # Batch size for training
        per_device_eval_batch_size=64,   # Batch size for evaluation
        warmup_steps=500,                # Number of warmup steps for learning rate
        weight_decay=0.01,               # Strength of weight decay (regularization)
        logging_dir='./logs',            # Directory for storing logs
        logging_steps=10,
        evaluation_strategy="epoch",     # Evaluate at the end of each epoch
    )

    # Initialize HuggingFace Trainer
    trainer = Trainer(
        model=model,                         # The model to train
        args=training_args,                  # Training arguments
        train_dataset=train_dataset,         # Training dataset
        eval_dataset=test_dataset            # Evaluation dataset
    )

    # Train the model
    print("Starting training... This might take a while depending on hardware.")
    trainer.train()

    # Save the trained model and tokenizer locally
    print(f"Training complete. Saving model to {model_save_path}...")
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print("Model saved successfully.")

def run_training_async(data_path: str, model_save_path: str):
    """
    Helper to run training in a background thread so the API doesn't block.
    """
    thread = threading.Thread(target=start_training, args=(data_path, model_save_path))
    thread.start()
