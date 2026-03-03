"""
Google Gemini AI integration for intelligent responses.
Provides advanced AI-powered answers to any question.
"""

import os
from google import genai
from google.genai import types
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class GeminiAI:
    def __init__(self):
        """Initialize Gemini AI with API key."""
        self.api_key = os.getenv('GEMINI_API_KEY')
        
        if not self.api_key or self.api_key == 'your-gemini-api-key-here':
            print("WARNING: Gemini API key not configured. Using fallback responses.")
            self.client = None
        else:
            try:
                self.client = genai.Client(api_key=self.api_key)
                print("Gemini AI initialized successfully!")
            except Exception as e:
                print(f"Error initializing Gemini AI: {e}")
                self.client = None
    
    def get_response(self, user_message: str, context: str = "", file_content: str = None) -> Optional[str]:
        """
        Get AI-powered response from Gemini.
        
        Args:
            user_message: User's question or message
            context: Conversation context (optional)
            file_content: Extracted content from uploaded file (optional)
            
        Returns:
            AI-generated response or None if unavailable
        """
        if not self.client:
            print("DEBUG: Gemini client not available")
            return None
        
        try:
            # Build prompt with file content and context
            if file_content:
                prompt = f"User uploaded a file with this content:\n\n{file_content}\n\n"
                if context:
                    prompt += f"Context: {context}\n\n"
                prompt += f"User question: {user_message}\n\nPlease analyze the file content and answer the question."
            elif context:
                prompt = f"Context: {context}\n\nUser: {user_message}\n\nAssistant:"
            else:
                prompt = user_message
            
            print(f"DEBUG: Calling Gemini with prompt: {prompt[:100]}...")
            
            # Generate response using new API
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            
            print(f"DEBUG: Gemini response received")
            
            if response and response.text:
                print(f"DEBUG: Returning Gemini text: {response.text[:100]}...")
                return response.text.strip()
            else:
                print("DEBUG: No text in Gemini response")
                return None
                
        except Exception as e:
            print(f"ERROR getting Gemini response: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def is_available(self) -> bool:
        """Check if Gemini AI is available."""
        return self.client is not None
