import os
import re
import requests
from typing import List

# Setup Google Generative AI (Gemini) - which offers a 100% free tier perfect for Hackathons!
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-2.5-flash") # The free, fast tier

def generate_explanation(text: str, prediction: str) -> List[str]:
    """
    Generates dynamic explanations using 100% Free APIs!
    1. Uses Google Gemini API (if key is set) for deep reasoning.
    2. Falls back to the free public Wikipedia REST API for real-time fact-checking.
    """
    # 1. First Attempt: Use the Free Gemini API if configured
    if GEMINI_API_KEY:
        try:
            prompt = f"Analyze this text: '{text}'. The AI predicted it is '{prediction}'. Briefly explain why in 2 bullet points, focusing on misinformation tactics or factual backing."
            response = gemini_model.generate_content(prompt)
            # Split the response into a list of points
            return [line.strip("* \n") for line in response.text.split("\n") if line.strip()][:3]
        except Exception as e:
            print(f"Gemini API Error: {e}")
            # Fall through to Wikipedia API

    # 2. Second Attempt: Use 100% open, NO-KEY Wikipedia API for dynamic context
    explanations = []
    
    # Naive entity extraction: pick the longest words as main subjects
    words = [w.strip() for w in re.split(r'\W+', text) if len(w) > 4]
    if not words:
        return ["Text too short to query free public APIs for context."]
        
    subjects = sorted(list(set(words)), key=len, reverse=True)[:2]
    
    for subject in subjects:
        try:
            # Query Wikipedia's free public REST API
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{subject}"
            response = requests.get(url, timeout=3)
            
            if response.status_code == 200:
                data = response.json()
                if "extract" in data:
                    # Append the dynamic real-world context found from Wikipedia!
                    summary = data["extract"].split(". ")[0] + "." 
                    explanations.append(f"Wikipedia Context for '{subject}': {summary}")
        except Exception:
            continue
            
    if not explanations:
        if prediction == "Potentially Misleading":
            explanations.append("The AI model determined the text context is similar to known misinformation.")
        else:
            explanations.append("No major suspicious patterns detected via public open data.")
            
    return explanations

def get_dynamic_fallback_score(text: str) -> tuple:
    """
    If the text contains verified Wikipedia entities, boost credibility.
    Returns (score, label, dynamic_explanations)
    """
    explanations = generate_explanation(text, "Fallback Check")
    
    if any("Wikipedia Context" in exp for exp in explanations):
        # We found real API data!
        return 75, "Potentially Credible", explanations
    else:
        return 35, "Potentially Misleading", ["Free API could not verify public encyclopedia records for these entities.", *explanations]
