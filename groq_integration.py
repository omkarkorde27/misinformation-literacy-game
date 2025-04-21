"""
Module for Groq API integration to provide explanations for fake news classification
"""

import requests
import json
import os
from typing import Dict, Any, List, Optional, Tuple

class GroqExplainer:
    """Class to provide explanations for fake news classification using Groq's API"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "llama3-70b-8192"):
        """
        Initialize the Groq API client
        
        Args:
            api_key: The Groq API key (if None, will look for GROQ_API_KEY env variable)
            model: The language model to use (default: llama3-70b-8192)
        """
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            print("Warning: No Groq API key provided. Set GROQ_API_KEY environment variable or pass api_key parameter.")
        
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        
    def get_explanation(self, text: str, classification: str, confidence: float, 
                        features: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Get an explanation for why the text was classified as real or fake news
        
        Args:
            text: The news text that was classified
            classification: The classification result ("Real News" or "Fake News")
            confidence: The confidence score (0-1) of the classification
            features: Optional dictionary of feature importance scores
            
        Returns:
            Dictionary with explanation and any additional analysis
        """
        if not self.api_key:
            return {"error": "No API key provided", "explanation": "API key required to generate explanation."}
        
        # Prepare the message for the API
        messages = self._create_prompt(text, classification, confidence, features)
        
        # Prepare the API request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,  # Low temperature for more consistent responses
            "max_tokens": 1024
        }
        
        try:
            # Make the API request
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Extract and format the explanation
            explanation = result["choices"][0]["message"]["content"]
            
            return {
                "explanation": explanation,
                "model_used": self.model,
                "status": "success"
            }
            
        except requests.exceptions.RequestException as e:
            print(f"Error making request to Groq API: {e}")
            return {
                "error": str(e),
                "explanation": "An error occurred while generating the explanation.",
                "status": "error"
            }
    
    def _create_prompt(self, text: str, classification: str, confidence: float, 
                      features: Optional[Dict[str, float]] = None) -> List[Dict[str, str]]:
        """
        Create the prompt for the Groq API
        
        Args:
            text: The news text that was classified
            classification: The classification result ("Real News" or "Fake News")
            confidence: The confidence score (0-1) of the classification
            features: Optional dictionary of feature importance scores
            
        Returns:
            List of message dictionaries for the API request
        """
        # Determine some general aspects of the classification
        confidence_level = "very high" if confidence > 0.9 else "high" if confidence > 0.8 else \
                          "moderate" if confidence > 0.65 else "low"
        
        # Format features information if provided
        features_text = ""
        if features:
            features_text = "Feature importance scores:\n"
            for feature, score in features.items():
                features_text += f"- {feature}: {score:.3f}\n"
        
        # Create the system message
        system_message = {
            "role": "system",
            "content": (
                "You are an expert in media literacy and fake news detection. Your task is to provide "
                "a clear, detailed explanation of why a news text was classified as real or fake by "
                "an AI model. Focus on specific elements in the text that could indicate its "
                f"authenticity or lack thereof. Be objective and educational in your response."
            )
        }
        
        # Create the user message
        user_message = {
            "role": "user",
            "content": (
                f"The following news text was classified as '{classification}' with {confidence:.1%} confidence "
                f"({confidence_level} confidence level).\n\n"
                f"Text: \"{text}\"\n\n"
                f"{features_text}\n"
                "Please provide:\n"
                "1. A detailed explanation of why this text was likely classified this way\n"
                "2. Specific elements or patterns in the text that support this classification\n"
                "3. What someone should look for to verify this information independently\n"
                "4. Keep your response concise (maximum 500 words) and educational"
            )
        }
        
        return [system_message, user_message]


# Example usage
if __name__ == "__main__":
    # Example usage with placeholder API key
    explainer = GroqExplainer(api_key="your-api-key-here")
    
    sample_text = "BREAKING: Scientists discover miracle cure for all diseases - Government hiding the truth!"
    classification = "Fake News"
    confidence = 0.92
    
    features = {
        "text_length": 0.2,
        "exclamation_count": 0.8,
        "question_count": 0.1,
        "uppercase_ratio": 0.7,
        "avg_word_length": 0.3,
        "clickbait_score": 0.9
    }
    
    explanation = explainer.get_explanation(sample_text, classification, confidence, features)
    
    print(json.dumps(explanation, indent=2))