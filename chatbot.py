"""
Main chatbot class that integrates NLP, RL agent, and feedback handling.
Provides the core conversational interface with learning capabilities.
"""

import uuid
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from nlp_processor import NLPProcessor
from rl_agent import QLearningAgent
from feedback_handler import FeedbackHandler
from database import DatabaseManager
from gemini_ai import GeminiAI

class IntelligentChatbot:
    def __init__(self, db_path: str = "data/conversations.db"):
        """Initialize the intelligent chatbot with all components."""
        self.nlp_processor = NLPProcessor()
        self.rl_agent = QLearningAgent()
        self.feedback_handler = FeedbackHandler()
        self.database = DatabaseManager(db_path)
        self.gemini_ai = GeminiAI()
        
        # Load existing Q-table if available
        self.rl_agent.load_q_table(self.database)
        
        # Session management
        self.active_sessions = {}
        
        print("Intelligent Chatbot initialized successfully!")
    
    def create_session(self) -> str:
        """Create a new conversation session."""
        session_id = str(uuid.uuid4())
        self.active_sessions[session_id] = {
            'created_at': datetime.now(),
            'message_count': 0,
            'last_conversation_id': None
        }
        return session_id
    
    def process_message(self, session_id: str, user_message: str) -> Dict:
        """
        Process user message and generate response.
        
        Args:
            session_id: Session identifier
            user_message: User's input message
            
        Returns:
            Dictionary containing bot response and metadata
        """
        # Validate session
        if session_id not in self.active_sessions:
            session_id = self.create_session()
        
        # Get conversation history
        conversation_history = self.database.get_conversation_history(session_id)
        
        # NLP Processing
        intent = self.nlp_processor.detect_intent(user_message)
        context = self.nlp_processor.extract_context(conversation_history)
        state = self.nlp_processor.get_state_representation(user_message, context, intent)
        
        # Try to get response from Gemini AI first
        selected_response = None
        if self.gemini_ai.is_available():
            selected_response = self.gemini_ai.get_response(user_message, context)
        
        # Fallback to traditional NLP if Gemini is unavailable
        if not selected_response:
            # Get candidate responses
            candidate_responses = self.nlp_processor.get_response_candidates(intent, user_message)
            
            # RL Agent selects best response
            selected_response = self.rl_agent.select_action(state, candidate_responses)
        
        # Save conversation to database
        conversation_id = self.database.save_conversation(
            session_id=session_id,
            user_message=user_message,
            bot_response=selected_response,
            context=context,
            intent=intent
        )
        
        # Process implicit feedback
        implicit_reward = self.feedback_handler.process_implicit_feedback(
            session_id, user_message, selected_response, conversation_history
        )
        
        # Update RL agent with implicit feedback
        if implicit_reward != 0:
            self.rl_agent.update_q_value(implicit_reward)
        
        # Update session info
        self.active_sessions[session_id]['message_count'] += 1
        self.active_sessions[session_id]['last_conversation_id'] = conversation_id
        
        # Check for conversation end signals
        is_ending = self.feedback_handler.detect_conversation_end_signals(user_message)
        if is_ending:
            end_reward = self.rl_agent.get_reward_from_feedback('end', 0)
            self.rl_agent.update_q_value(end_reward)
        
        return {
            'response': selected_response,
            'conversation_id': conversation_id,
            'session_id': session_id,
            'intent': intent,
            'state': state,
            'confidence': self._calculate_response_confidence(state, selected_response),
            'is_ending': is_ending,
            'metadata': {
                'message_count': self.active_sessions[session_id]['message_count'],
                'implicit_reward': implicit_reward,
                'q_stats': self.rl_agent.get_q_table_stats(),
                'ai_powered': self.gemini_ai.is_available()
            }
        }
    
    def process_explicit_feedback(self, conversation_id: int, feedback_type: str, 
                                feedback_value: Optional[float] = None) -> Dict:
        """
        Process explicit user feedback and update learning.
        
        Args:
            conversation_id: ID of the conversation being rated
            feedback_type: Type of feedback ('like', 'dislike', 'rating')
            feedback_value: Numerical value for ratings
            
        Returns:
            Processing result and updated statistics
        """
        # Process feedback and get reward
        reward = self.feedback_handler.process_explicit_feedback(
            conversation_id, feedback_type, feedback_value
        )
        
        # Save feedback to database
        self.database.save_feedback(conversation_id, feedback_type, feedback_value or 0)
        
        # Update RL agent
        self.rl_agent.update_q_value(reward)
        
        # Save updated Q-table
        self.rl_agent.save_q_table(self.database)
        
        return {
            'feedback_processed': True,
            'reward': reward,
            'feedback_type': feedback_type,
            'q_stats': self.rl_agent.get_q_table_stats(),
            'message': f"Thank you for your feedback! I'll use this to improve my responses."
        }
    
    def get_session_stats(self, session_id: str) -> Dict:
        """Get statistics for a specific session."""
        if session_id not in self.active_sessions:
            return {'error': 'Session not found'}
        
        conversation_history = self.database.get_conversation_history(session_id)
        session_info = self.active_sessions[session_id]
        
        return {
            'session_id': session_id,
            'created_at': session_info['created_at'].isoformat(),
            'message_count': session_info['message_count'],
            'conversation_length': len(conversation_history),
            'intents_detected': list(set(turn.get('intent', 'unknown') for turn in conversation_history)),
            'last_activity': conversation_history[-1]['timestamp'] if conversation_history else None
        }
    
    def get_learning_stats(self) -> Dict:
        """Get comprehensive learning statistics."""
        q_stats = self.rl_agent.get_q_table_stats()
        feedback_stats = self.database.get_feedback_stats()
        feedback_summary = self.feedback_handler.get_feedback_summary()
        
        return {
            'q_learning': q_stats,
            'feedback_database': feedback_stats,
            'feedback_analysis': feedback_summary,
            'improvement_suggestions': self.feedback_handler.suggest_improvements(feedback_summary),
            'total_sessions': len(self.active_sessions),
            'learning_history_length': len(self.rl_agent.learning_history)
        }
    
    def _calculate_response_confidence(self, state: str, response: str) -> float:
        """Calculate confidence score for the selected response."""
        if state not in self.rl_agent.q_table:
            return 0.5  # Neutral confidence for new states
        
        state_actions = self.rl_agent.q_table[state]
        if response not in state_actions:
            return 0.3  # Low confidence for new actions
        
        response_q_value = state_actions[response]
        
        # Normalize Q-value to confidence score (0-1)
        all_q_values = list(state_actions.values())
        if len(all_q_values) <= 1:
            return 0.5
        
        min_q = min(all_q_values)
        max_q = max(all_q_values)
        
        if max_q == min_q:
            return 0.5
        
        # Normalize to 0-1 range
        confidence = (response_q_value - min_q) / (max_q - min_q)
        return max(0.1, min(0.9, confidence))  # Clamp between 0.1 and 0.9
    
    def get_best_responses_for_intent(self, intent: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Get best learned responses for a given intent."""
        # Find states that match the intent
        matching_states = [
            state for state in self.rl_agent.q_table.keys()
            if state.startswith(intent)
        ]
        
        if not matching_states:
            return []
        
        # Collect all responses for this intent with their average Q-values
        response_scores = {}
        for state in matching_states:
            for response, q_value in self.rl_agent.q_table[state].items():
                if response not in response_scores:
                    response_scores[response] = []
                response_scores[response].append(q_value)
        
        # Calculate average Q-values
        avg_response_scores = {
            response: sum(scores) / len(scores)
            for response, scores in response_scores.items()
        }
        
        # Sort and return top-k
        sorted_responses = sorted(
            avg_response_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_responses[:top_k]
    
    def reset_learning(self):
        """Reset the learning state (for testing/evaluation purposes)."""
        self.rl_agent = QLearningAgent()
        self.feedback_handler = FeedbackHandler()
        print("Learning state has been reset.")
    
    def export_learning_data(self) -> Dict:
        """Export learning data for analysis."""
        return {
            'q_table': dict(self.rl_agent.q_table),
            'learning_history': self.rl_agent.learning_history,
            'feedback_history': self.feedback_handler.feedback_history,
            'visit_counts': dict(self.rl_agent.visit_counts),
            'performance_stats': self.get_learning_stats()
        }