"""
Google Gemini AI integration for intelligent responses.
Provides advanced AI-powered answers to any question.
"""

import os
import google.generativeai as genai
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class GeminiAI:
    def __init__(self):
        """Initialize Gemini AI with API key."""
        self.api_key = os.getenv('GEMINI_API_KEY')
        
        if not self.api_key or self.api_key == 'your-gemini-api-key-here':
            print("WARNING: Gemini API key not configured. Using fallback responses.")
            self.model = None
        else:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-pro')
                print("Gemini AI initialized successfully!")
            except Exception as e:
                print(f"Error initializing Gemini AI: {e}")
                self.model = None
    
    def get_response(self, user_message: str, context: str = "") -> Optional[str]:
        """
        Get AI-powered response from Gemini.
        
        Args:
            user_message: User's question or message
            context: Conversation context (optional)
            
        Returns:
            AI-generated response or None if unavailable
        """
        if not self.model:
            return None
        
        try:
            # Build prompt with context if available
            if context:
                prompt = f"Context: {context}\n\nUser: {user_message}\n\nAssistant:"
            else:
                prompt = user_message
            
            # Generate response
            response = self.model.generate_content(prompt)
            
            if response and response.text:
                return response.text.strip()
            else:
                return None
                
        except Exception as e:
            print(f"Error getting Gemini response: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if Gemini AI is available."""
        return self.model is not None
