"""
Natural Language Processing utilities for the chatbot.
Handles tokenization, vectorization, intent detection, and context tracking.
"""

import re
import nltk
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional import for sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class NLPProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Load sentence transformer for better embeddings
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            except:
                self.sentence_model = None
                print("Warning: Could not load sentence transformer, using TF-IDF only")
        else:
            self.sentence_model = None
            print("Info: Using TF-IDF vectorization (sentence-transformers not installed)")
        
        # Enhanced intents and keywords with topic detection
        self.intent_keywords = {
            'greeting': ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'greetings'],
            'goodbye': ['bye', 'goodbye', 'see you', 'farewell', 'take care', 'later', 'exit'],
            'question_ai': ['artificial intelligence', 'ai', 'machine learning', 'ml', 'deep learning', 'neural network', 'algorithm'],
            'question_tech': ['technology', 'computer', 'programming', 'software', 'coding', 'development', 'tech'],
            'question_science': ['science', 'physics', 'chemistry', 'biology', 'mathematics', 'research'],
            'question_general': ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'explain', 'tell me'],
            'help': ['help', 'assist', 'support', 'guide', 'explain', 'show me', 'teach'],
            'thanks': ['thank', 'thanks', 'appreciate', 'grateful', 'thx'],
            'complaint': ['bad', 'terrible', 'awful', 'hate', 'dislike', 'wrong', 'error', 'problem'],
            'compliment': ['good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'awesome', 'perfect'],
            'request': ['can you', 'could you', 'please', 'would you', 'i need', 'i want', 'show me'],
            'personal': ['how are you', 'who are you', 'your name', 'about you', 'tell me about yourself']
        }
        
        # Comprehensive knowledge base for intelligent responses
        self.knowledge_base = {
            # AI and Machine Learning Topics
            'artificial intelligence': "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, language understanding, and decision-making. AI systems can be narrow (designed for specific tasks) or general (capable of performing any intellectual task that humans can do).",
            
            'machine learning': "Machine Learning (ML) is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. ML algorithms build mathematical models based on training data to make predictions or decisions. There are three main types: supervised learning (learning with labeled examples), unsupervised learning (finding patterns in unlabeled data), and reinforcement learning (learning through interaction and feedback).",
            
            'deep learning': "Deep Learning is a subset of machine learning that uses artificial neural networks with multiple layers (hence 'deep') to model and understand complex patterns in data. It's inspired by the structure and function of the human brain. Deep learning is particularly effective for tasks like image recognition, natural language processing, speech recognition, and game playing. Popular architectures include convolutional neural networks (CNNs) and recurrent neural networks (RNNs).",
            
            'neural network': "A Neural Network is a computing system inspired by biological neural networks that constitute animal brains. It consists of interconnected nodes (artificial neurons) organized in layers. Each connection has a weight that adjusts as learning proceeds. Neural networks can learn to recognize patterns, classify data, and make predictions by processing examples and adjusting their internal parameters through training algorithms like backpropagation.",
            
            'reinforcement learning': "Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by performing actions in an environment to maximize cumulative reward. The agent learns through trial and error, receiving feedback in the form of rewards or penalties. Key concepts include states, actions, rewards, and policies. RL is used in game playing (like AlphaGo), robotics, autonomous vehicles, and recommendation systems. I myself use reinforcement learning to improve my responses based on your feedback!",
            
            'algorithm': "An algorithm is a step-by-step procedure or set of rules designed to solve a specific problem or perform a particular task. In computer science, algorithms are the foundation of all programs and software. They can be simple (like sorting a list) or complex (like machine learning algorithms). Good algorithms are efficient, correct, and clear. Examples include search algorithms, sorting algorithms, and optimization algorithms.",
            
            # Technology and Programming
            'programming': "Programming is the process of creating instructions for computers using programming languages. It involves writing code to solve problems, create applications, and automate tasks. Programming requires logical thinking, problem-solving skills, and understanding of programming concepts like variables, functions, loops, and data structures. Popular programming languages include Python, JavaScript, Java, C++, and many others, each with their own strengths and use cases.",
            
            'software': "Software refers to computer programs and applications that run on hardware systems. It includes everything from operating systems (like Windows, macOS, Linux) to mobile apps, web applications, games, and productivity tools. Software is created through programming and can be system software (managing computer resources) or application software (performing specific tasks for users). Modern software development involves planning, coding, testing, and maintenance.",
            
            'computer': "A computer is an electronic device that processes data according to instructions (programs). Modern computers consist of hardware components like the CPU (processor), memory (RAM), storage (hard drives/SSDs), and input/output devices. They can perform complex calculations, store vast amounts of information, connect to networks, and run multiple programs simultaneously. Computers have revolutionized nearly every aspect of modern life.",
            
            'technology': "Technology refers to the application of scientific knowledge for practical purposes, especially in industry and everyday life. In computing, technology encompasses hardware, software, networks, and digital systems. Modern technology includes smartphones, internet, cloud computing, artificial intelligence, robotics, and biotechnology. Technology continues to evolve rapidly, transforming how we work, communicate, learn, and live.",
            
            'coding': "Coding, also known as programming, is the process of writing instructions for computers using programming languages. It involves translating human ideas and logic into a language that computers can understand and execute. Coding requires learning syntax (the rules of a programming language), logic, problem-solving, and debugging skills. Coders create everything from simple scripts to complex applications, websites, games, and AI systems.",
            
            'development': "Software development is the process of creating, designing, deploying, and maintaining software applications. It involves several phases: planning and analysis, design, implementation (coding), testing, deployment, and maintenance. Development can follow different methodologies like Agile, Waterfall, or DevOps. Modern development often involves teams of developers, designers, testers, and project managers working together to create high-quality software products.",
            
            # Science and Mathematics
            'science': "Science is the systematic study of the natural world through observation, experimentation, and analysis. It seeks to understand how things work and why they behave as they do. Major branches include physics (matter and energy), chemistry (substances and reactions), biology (living organisms), earth science (our planet), and astronomy (space and celestial objects). Science drives technological advancement and helps us solve real-world problems.",
            
            'mathematics': "Mathematics is the study of numbers, shapes, patterns, and logical reasoning. It provides the foundation for science, engineering, technology, and many other fields. Key areas include arithmetic, algebra, geometry, calculus, statistics, and discrete mathematics. In computer science, mathematics is essential for algorithms, cryptography, graphics, machine learning, and data analysis. Mathematical thinking develops problem-solving and analytical skills.",
            
            'physics': "Physics is the fundamental science that studies matter, energy, motion, forces, space, and time. It seeks to understand how the universe works at all scales, from subatomic particles to galaxies. Physics principles underlie all other sciences and drive technological innovations. Key areas include mechanics, thermodynamics, electromagnetism, quantum mechanics, and relativity. Physics concepts are crucial in engineering, computer science, and technology development.",
            
            # Chatbot and Personal Information
            'chatbot': "A chatbot is a computer program designed to simulate conversation with human users through text or voice interactions. Chatbots use natural language processing (NLP) to understand user input and generate appropriate responses. They can be rule-based (following predefined scripts) or AI-powered (using machine learning). Modern chatbots are used for customer service, virtual assistants, education, entertainment, and information retrieval. I'm an example of an AI-powered chatbot that learns from user feedback!",
            
            'about_me': "I'm an intelligent AI chatbot powered by reinforcement learning technology. Unlike traditional chatbots that follow fixed scripts, I learn and improve from every conversation through your feedback. When you give me likes, dislikes, or ratings, you're training my neural network to provide better responses over time. I can help with questions about artificial intelligence, technology, science, programming, and general topics. My goal is to become more helpful and accurate through our interactions!",
            
            'capabilities': "I can assist you with a wide range of topics including artificial intelligence, machine learning, programming, technology, science, and general questions. My key capabilities include: providing detailed explanations of complex topics, learning from your feedback to improve my responses, maintaining conversation context, detecting your intent and providing relevant information, and adapting my communication style based on your preferences. I'm particularly knowledgeable about AI and technology topics, and I continuously learn from our interactions!",
            
            'how_i_learn': "I learn through reinforcement learning, a type of machine learning where I improve based on feedback rewards. When you give me positive feedback (likes or high ratings), it reinforces good responses in my neural network. Negative feedback helps me avoid similar responses in the future. This process is similar to how humans learn - through trial, error, and feedback. Each conversation teaches me something new about what responses are helpful, accurate, and engaging. Your feedback is crucial for my learning process!",
            
            # General Knowledge
            'internet': "The Internet is a global network of interconnected computers that communicate using standardized protocols. It enables the sharing of information, communication, and access to services worldwide. The Internet supports various applications like the World Wide Web, email, file sharing, streaming, and social media. It has revolutionized commerce, education, entertainment, and social interaction. The Internet infrastructure includes servers, routers, cables, and wireless networks that span the globe.",
            
            'data': "Data refers to facts, statistics, or information collected for analysis or reference. In computing, data can be text, numbers, images, audio, video, or any digital information. Data is processed by computers to generate useful information and insights. Key concepts include data types, data structures, databases, data analysis, and data science. In the age of big data and AI, the ability to collect, store, process, and analyze large amounts of data has become crucial for businesses and research."
        }
        
        # Initialize with some sample texts for TF-IDF
        sample_texts = [
            "Hello how are you today",
            "What is the weather like",
            "Can you help me with something",
            "Thank you for your assistance",
            "Goodbye see you later"
        ]
        self.tfidf_vectorizer.fit(sample_texts)
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text and remove stop words."""
        text = self.preprocess_text(text)
        tokens = word_tokenize(text)
        
        # Remove stop words
        tokens = [token for token in tokens if token not in self.stop_words]
        
        return tokens
    
    def detect_intent(self, text: str) -> str:
        """Detect user intent from text with enhanced topic detection."""
        text_lower = text.lower()
        
        # Score each intent based on keyword matches
        intent_scores = {}
        for intent, keywords in self.intent_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    # Give higher weight to exact matches
                    if keyword == text_lower.strip():
                        score += 3
                    elif len(keyword.split()) > 1:  # Multi-word phrases get higher weight
                        score += 2
                    else:
                        score += 1
            
            if score > 0:
                intent_scores[intent] = score
        
        # Return intent with highest score, or 'question_general' if no matches
        if intent_scores:
            return max(intent_scores, key=intent_scores.get)
        else:
            return 'question_general'
    
    def get_intelligent_response(self, user_message: str, intent: str, context: str = "") -> str:
        """Generate intelligent, contextual responses based on message content."""
        text_lower = user_message.lower().strip()
        
        # Direct keyword matching for better accuracy
        response_keywords = {
            'artificial intelligence': ['artificial intelligence', 'ai is', 'what is ai', 'define ai', 'ai means'],
            'machine learning': ['machine learning', 'ml is', 'what is ml', 'define machine learning', 'machine learning means'],
            'deep learning': ['deep learning', 'what is deep learning', 'define deep learning', 'deep learning means'],
            'neural network': ['neural network', 'what is neural network', 'define neural network', 'neural networks'],
            'reinforcement learning': ['reinforcement learning', 'what is reinforcement learning', 'rl is', 'define reinforcement learning'],
            'algorithm': ['algorithm', 'what is algorithm', 'define algorithm', 'algorithms'],
            'programming': ['programming', 'what is programming', 'define programming', 'coding is'],
            'software': ['software', 'what is software', 'define software'],
            'computer': ['computer', 'what is computer', 'define computer'],
            'technology': ['technology', 'what is technology', 'define technology'],
            'coding': ['coding', 'what is coding', 'define coding'],
            'development': ['development', 'software development', 'what is development'],
            'science': ['science', 'what is science', 'define science'],
            'mathematics': ['mathematics', 'math', 'what is mathematics', 'define mathematics'],
            'physics': ['physics', 'what is physics', 'define physics'],
            'chatbot': ['chatbot', 'what is chatbot', 'define chatbot', 'what is a chatbot'],
            'internet': ['internet', 'what is internet', 'define internet'],
            'data': ['data', 'what is data', 'define data']
        }
        
        # Check for direct matches first
        for topic, keywords in response_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if topic in self.knowledge_base:
                        return self.knowledge_base[topic]
        
        # Check for partial matches and related terms
        if any(term in text_lower for term in ['artificial intelligence', 'ai']):
            return self.knowledge_base['artificial intelligence']
        elif any(term in text_lower for term in ['machine learning', 'ml']):
            return self.knowledge_base['machine learning']
        elif 'deep learning' in text_lower:
            return self.knowledge_base['deep learning']
        elif any(term in text_lower for term in ['neural network', 'neural networks']):
            return self.knowledge_base['neural network']
        elif 'reinforcement learning' in text_lower:
            return self.knowledge_base['reinforcement learning']
        elif 'algorithm' in text_lower:
            return self.knowledge_base['algorithm']
        elif 'programming' in text_lower or 'coding' in text_lower:
            return self.knowledge_base['programming']
        elif 'software' in text_lower:
            return self.knowledge_base['software']
        elif 'computer' in text_lower:
            return self.knowledge_base['computer']
        elif 'technology' in text_lower:
            return self.knowledge_base['technology']
        elif 'development' in text_lower:
            return self.knowledge_base['development']
        elif 'science' in text_lower:
            return self.knowledge_base['science']
        elif any(term in text_lower for term in ['mathematics', 'math']):
            return self.knowledge_base['mathematics']
        elif 'physics' in text_lower:
            return self.knowledge_base['physics']
        elif 'chatbot' in text_lower:
            return self.knowledge_base['chatbot']
        elif 'internet' in text_lower:
            return self.knowledge_base['internet']
        elif 'data' in text_lower:
            return self.knowledge_base['data']
        
        # Handle personal questions
        if any(phrase in text_lower for phrase in ['who are you', 'about you', 'your name', 'tell me about yourself']):
            return self.knowledge_base['about_me']
        elif any(phrase in text_lower for phrase in ['what can you do', 'capabilities', 'help with', 'your capabilities']):
            return self.knowledge_base['capabilities']
        elif any(phrase in text_lower for phrase in ['how do you learn', 'how you learn', 'learning process']):
            return self.knowledge_base['how_i_learn']
        
        # No specific match found
        return None
    
    def vectorize_text(self, text: str) -> np.ndarray:
        """Convert text to vector representation."""
        if self.sentence_model:
            # Use sentence transformer for better embeddings
            return self.sentence_model.encode([text])[0]
        else:
            # Fallback to TF-IDF
            try:
                tfidf_vector = self.tfidf_vectorizer.transform([text])
                return tfidf_vector.toarray()[0]
            except:
                # If text contains unknown words, return zero vector
                return np.zeros(1000)
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        vec1 = self.vectorize_text(text1)
        vec2 = self.vectorize_text(text2)
        
        # Reshape for cosine similarity calculation
        vec1 = vec1.reshape(1, -1)
        vec2 = vec2.reshape(1, -1)
        
        similarity = cosine_similarity(vec1, vec2)[0][0]
        return similarity
    
    def extract_context(self, conversation_history: List[Dict]) -> str:
        """Extract context from conversation history."""
        if not conversation_history:
            return ""
        
        # Get last few messages for context
        recent_messages = conversation_history[-3:]
        
        context_parts = []
        for msg in recent_messages:
            context_parts.append(f"User: {msg['user_message']}")
            context_parts.append(f"Bot: {msg['bot_response']}")
        
        return " | ".join(context_parts)
    
    def get_state_representation(self, user_message: str, context: str, intent: str) -> str:
        """Create a state representation for RL agent."""
        # Combine intent and key features of the message
        tokens = self.tokenize(user_message)
        key_words = tokens[:3]  # Take first 3 meaningful words
        
        # Create a compact state representation
        state = f"{intent}_{'-'.join(key_words)}"
        
        # Add context indicator if available
        if context:
            context_intent = self.detect_intent(context.split('|')[-1] if '|' in context else context)
            state += f"_ctx_{context_intent}"
        
        return state
    
    def get_response_candidates(self, intent: str, user_message: str = "") -> List[str]:
        """Get candidate responses based on intent with intelligent content."""
        
        # First try to get intelligent response
        intelligent_response = self.get_intelligent_response(user_message, intent)
        if intelligent_response:
            return [intelligent_response]
        
        # Enhanced response templates with more professional and informative content
        response_templates = {
            'greeting': [
                "Hello! I'm an AI assistant powered by reinforcement learning. I specialize in explaining complex topics like artificial intelligence, technology, and science. How can I help you today?",
                "Hi there! I'm an intelligent chatbot that learns from your feedback to provide better responses. I can help with AI, programming, technology, and general questions. What would you like to know?",
                "Good day! I'm an AI assistant that uses machine learning to improve through our conversations. I'm particularly knowledgeable about technology and science topics. How may I assist you?",
                "Hello! Nice to meet you. I'm an AI chatbot that learns from user feedback to become more helpful over time. I can explain complex topics in simple terms. What interests you?"
            ],
            'goodbye': [
                "Goodbye! Thank you for helping me learn through our conversation. Your feedback helps improve my responses for future users. Have a great day!",
                "See you later! I appreciate the opportunity to learn from our interaction. Your feedback is valuable for my continuous improvement. Take care!",
                "Farewell! It was a pleasure discussing topics with you and learning from your feedback. Feel free to return anytime with more questions!",
                "Bye! Thanks for the engaging conversation. Remember, each interaction helps me become a better AI assistant. Come back anytime!"
            ],
            'question_ai': [
                "That's an excellent question about artificial intelligence! AI is a fascinating field that encompasses machine learning, neural networks, and intelligent systems. Let me provide you with a detailed explanation.",
                "Great question about AI and machine learning! These technologies are transforming our world in incredible ways. I'd be happy to break down the concepts for you.",
                "Wonderful AI-related question! As an AI system myself, I'm particularly passionate about explaining these concepts clearly and accurately.",
                "Excellent question about artificial intelligence! This is one of the most exciting fields in technology today, with applications ranging from healthcare to autonomous vehicles."
            ],
            'question_tech': [
                "That's a great technology question! Technology is constantly evolving and shaping our digital world. Let me share some detailed insights about that topic.",
                "Excellent tech question! Understanding technology concepts is crucial in today's digital age. I'll provide you with a comprehensive explanation.",
                "Good question about technology! From programming to software development, technology drives innovation across all industries. Here's what you should know.",
                "Interesting technical question! Technology encompasses everything from basic computing to advanced systems. Let me explain that concept clearly."
            ],
            'question_science': [
                "That's a fascinating science question! Science helps us understand the natural world and drives technological advancement. Let me explain that concept in detail.",
                "Excellent scientific question! Science forms the foundation for many technological innovations we see today. I'd be happy to provide a thorough explanation.",
                "Great question about science! Scientific principles underlie much of our modern technology and understanding of the world. Here's a detailed answer.",
                "Wonderful science question! From physics to mathematics, scientific knowledge is essential for technological progress. Let me break that down for you."
            ],
            'question_general': [
                "That's an interesting question! I'll do my best to provide you with accurate and helpful information. Let me think about the most comprehensive way to explain that.",
                "Good question! I enjoy helping people understand complex topics by breaking them down into clear, understandable explanations. Here's what I know about that.",
                "Excellent question! I'm designed to provide detailed, informative responses that help you learn. Let me share what I know about that topic.",
                "That's worth exploring! I aim to give thorough, educational responses that are both accurate and easy to understand. Here's my explanation."
            ],
            'help': [
                "I'm here to help! I specialize in explaining artificial intelligence, machine learning, programming, technology, science, and general topics. I can break down complex concepts into understandable explanations. What specific topic would you like to learn about?",
                "Of course! I'd be delighted to assist you. My expertise includes AI and machine learning concepts, programming and software development, technology trends, and scientific principles. I learn from your feedback to provide better explanations. What can I help you understand?",
                "I'll do my best to help you! I'm particularly good at explaining technical concepts in simple terms. Whether it's about artificial intelligence, programming, technology, or science, I can provide detailed explanations. What would you like to know?",
                "Absolutely! I'm designed to be a helpful educational assistant. I can explain complex topics like AI, machine learning, programming, and technology in ways that are easy to understand. What subject interests you?"
            ],
            'thanks': [
                "You're very welcome! I'm glad I could provide helpful information. Your positive feedback helps me learn which explanations are most effective. Feel free to ask more questions anytime!",
                "My pleasure! I enjoy sharing knowledge and helping people understand complex topics. Your appreciation motivates me to continue improving my responses through learning.",
                "I'm happy I could help! Providing clear, accurate explanations is what I'm designed for. Your feedback helps me become better at explaining concepts. Is there anything else you'd like to know?",
                "You're welcome! I love helping people learn about technology, AI, and science. Your positive response tells me I'm on the right track with my explanations. What else can I help you understand?"
            ],
            'complaint': [
                "I apologize that my response wasn't helpful or accurate. Your feedback is incredibly valuable - it helps me learn what doesn't work so I can improve. Could you let me know what specific information you were looking for?",
                "I'm sorry my explanation didn't meet your expectations. Negative feedback is actually very important for my learning process - it helps me understand where I need to improve. What would have been more helpful?",
                "Thank you for the honest feedback, even though it's critical. This helps me learn and provide better responses in the future. I'm designed to improve through both positive and negative feedback. How can I better assist you?",
                "I understand your frustration, and I appreciate you taking the time to provide feedback. This criticism helps me learn what responses aren't working well. Could you help me understand what information would be more useful?"
            ],
            'compliment': [
                "Thank you so much! Positive feedback like yours is crucial for my reinforcement learning system. It helps me understand which types of explanations and responses are most helpful and engaging.",
                "That's wonderful to hear! Your encouragement helps me learn which responses work well. This positive reinforcement strengthens my ability to provide similar high-quality explanations in the future.",
                "I really appreciate that feedback! Compliments help me understand what makes responses valuable and informative. This positive signal helps improve my learning algorithm.",
                "Thank you for the kind words! Your positive feedback is essential for my learning process. It helps me identify successful response patterns and continue improving my explanations."
            ],
            'request': [
                "I'd be happy to help with that request! I can provide detailed explanations about AI, technology, programming, science, and many other topics. What specific information are you looking for?",
                "Absolutely! I'm here to provide informative, educational responses. Whether you need explanations about technical concepts, scientific principles, or general knowledge, I'll do my best to help. What would you like to know?",
                "Sure thing! I specialize in breaking down complex topics into understandable explanations. From artificial intelligence to programming to general science, I can help clarify concepts. What's your question?",
                "Of course! I'm designed to be a helpful educational resource. I can explain technical concepts, provide detailed information, and help you understand complex topics. How can I assist you today?"
            ],
            'personal': [
                "I'm an AI chatbot powered by reinforcement learning technology! I learn from user feedback to continuously improve my responses. I specialize in explaining AI, technology, programming, and science topics in clear, understandable ways.",
                "I'm an intelligent assistant that uses machine learning to get better through our conversations! When you give me feedback, you're actually training my neural network. I love helping people understand complex topics like artificial intelligence and technology.",
                "I'm a learning AI system that improves through interaction and feedback! Each conversation teaches me something new about providing helpful explanations. I'm particularly knowledgeable about AI, programming, technology, and science.",
                "I'm an AI assistant that uses reinforcement learning to become more helpful over time! Your feedback - whether positive or negative - helps train my responses. I enjoy explaining complex concepts in simple, accessible terms."
            ]
        }
        
        return response_templates.get(intent, response_templates['question_general'])