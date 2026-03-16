import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from utils.preprocessing import preprocess_text
from utils.explanation import generate_explanation
import shap
import numpy as np
import scipy as sp

class TruthLensInference:
    def __init__(self, model_path: str):
        """
        Initializes the inference pipeline.
        Loads the fine-tuned model and tokenizer from the specified path.
        """
        self.model_path = model_path
        print(f"Loading TruthLens model from {model_path}...")
        
        try:
            # We use DistilBert for inference. It is lightweight and fast for latency-sensitive tasks.
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
            self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
            self.model.eval()
            print("Model loaded successfully.")
            
            # Initialize SHAP explainer for the transformer model
            print("Initializing SHAP explainer...")
            
            # SHAP requires a prediction function that outputs probabilities as numpy arrays
            def f(x):
                tv = torch.tensor([self.tokenizer.encode(v, padding="max_length", max_length=128, truncation=True) for v in x])
                outputs = self.model(tv)[0].detach().cpu().numpy()
                scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
                # We return the probability of class 1 (Misleading)
                val = sp.special.logit(scores[:,1])
                return val
                
            self.explainer = shap.Explainer(f, self.tokenizer)
            print("SHAP explainer ready.")
            self.is_loaded = True
        except Exception as e:
            print(f"Warning: Model could not be loaded from {model_path}. You might need to train the model first.")
            print(f"Error: {e}")
            self.is_loaded = False
            
    def predict(self, raw_text: str):
        """
        The prediction pipeline:
        Input -> Preprocessing -> Tokenizer -> DistilBERT Model -> Probability Output
        """
        
        # 1. Preprocessing
        clean_text = preprocess_text(raw_text)
        
        if not self.is_loaded:
            # Fallback for Hackathon testing if the model isn't trained yet
            return {
                "prediction": "Model Not Loaded",
                "credibility_score": 0,
                "explanation": ["Model weights not found. Please run the training pipeline first or provide a valid MODEL_PATH."]
            }
            
        # 2. Tokenizer
        inputs = self.tokenizer(clean_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        
        # 3. DistilBERT Model Forward Pass
        # We disable gradient calculation for faster inference since we aren't training here
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # 4. Probability Output
        # The model returns raw logits. We apply Softmax to convert them to probabilities (0-1).
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Extract probabilities for class 0 (Credible) and class 1 (Misleading)
        prob_credible = probabilities[0][0].item()
        prob_misleading = probabilities[0][1].item()
        
        # Calculate Credibility Score (0-100)
        # Scaled based on the probability of being credible.
        credibility_score = int(prob_credible * 100)
        
        # Determine strict prediction label
        if prob_misleading > 0.5:
            prediction_label = "Potentially Misleading"
        else:
            prediction_label = "Credible"
            
        # Generate SHAP values for true AI explainability
        try:
            shap_values = self.explainer([clean_text])
            
            # Extract the tokens and their corresponding SHAP values for this prediction
            tokens = shap_values.data[0]
            values = shap_values.values[0]
            
            # We want to find the words that contributed the most to the prediction
            word_impacts = list(zip(tokens, values))
            # Sort by absolute impact (highest magnitude first)
            word_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Filter out generic subwords/spacing and keep the top 3 impactful words
            top_words = []
            for word, impact in word_impacts:
                clean_word = word.strip()
                if clean_word and len(clean_word) > 2 and clean_word not in ["[CLS]", "[SEP]"]:
                    # If impact is POSITIVE, it pushed the model towards "Misleading"
                    # If impact is NEGATIVE, it pushed the model towards "Credible"
                    direction = "misleading" if impact > 0 else "credible"
                    top_words.append(f"The word '{clean_word}' strongly suggested the text was {direction} (Impact: {impact:.2f})")
                if len(top_words) >= 3:
                    break
                    
            explanation = top_words if top_words else ["SHAP analysis found no strongly weighted keywords."]
        except Exception as e:
            print(f"SHAP explanation failed: {e}")
            # Fallback to dynamic APIs if SHAP fails
            explanation = generate_explanation(clean_text, prediction_label)
        
        return {
            "prediction": prediction_label,
            "credibility_score": credibility_score,
            "explanation": explanation
        }

# Singleton instance for the API
_inference_instance = None

def get_inference(model_path: str) -> TruthLensInference:
    global _inference_instance
    if _inference_instance is None:
        _inference_instance = TruthLensInference(model_path)
    return _inference_instance
