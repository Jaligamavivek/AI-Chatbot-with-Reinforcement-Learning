"""
User feedback processing and reward calculation for the chatbot.
Handles different types of user feedback and converts them to learning signals.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np

class FeedbackHandler:
    def __init__(self):
        """Initialize feedback handler with tracking mechanisms."""
        self.feedback_history = []
        self.conversation_metrics = {}
        self.response_quality_tracker = {}
        
        # Feedback type weights for reward calculation
        self.feedback_weights = {
            'explicit_like': 1.0,
            'explicit_dislike': -1.0,
            'rating': 1.0,  # Will be scaled based on rating value
            'implicit_continue': 0.1,
            'implicit_end': -0.1,
            'correction': -0.5,
            'repetition': -0.3,
            'relevance': 0.5
        }
    
    def process_explicit_feedback(self, conversation_id: int, feedback_type: str, 
                                feedback_value: Optional[float] = None) -> float:
        """
        Process explicit user feedback (likes, dislikes, ratings).
        
        Args:
            conversation_id: ID of the conversation
            feedback_type: Type of feedback ('like', 'dislike', 'rating')
            feedback_value: Numerical value (for ratings)
            
        Returns:
            Reward value for RL agent
        """
        timestamp = datetime.now()
        
        if feedback_type == 'like':
            reward = self.feedback_weights['explicit_like']
            
        elif feedback_type == 'dislike':
            reward = self.feedback_weights['explicit_dislike']
            
        elif feedback_type == 'rating' and feedback_value is not None:
            # Scale rating (1-5) to reward (-1 to 1)
            normalized_rating = (feedback_value - 3.0) / 2.0
            reward = normalized_rating * self.feedback_weights['rating']
            
        else:
            reward = 0.0
        
        # Store feedback
        feedback_entry = {
            'conversation_id': conversation_id,
            'type': feedback_type,
            'value': feedback_value,
            'reward': reward,
            'timestamp': timestamp
        }
        
        self.feedback_history.append(feedback_entry)
        
        return reward
    
    def process_implicit_feedback(self, session_id: str, user_message: str, 
                                bot_response: str, conversation_history: List[Dict]) -> float:
        """
        Process implicit feedback signals from user behavior.
        
        Args:
            session_id: Session identifier
            user_message: Current user message
            bot_response: Bot's response
            conversation_history: Previous conversation turns
            
        Returns:
            Implicit reward value
        """
        total_reward = 0.0
        
        # 1. Conversation continuation reward
        if len(conversation_history) > 0:
            total_reward += self.feedback_weights['implicit_continue']
        
        # 2. Check for user corrections or clarifications
        correction_indicators = [
            'no, i meant', 'actually', 'that\'s not what i asked',
            'wrong', 'incorrect', 'not what i wanted', 'let me clarify'
        ]
        
        if any(indicator in user_message.lower() for indicator in correction_indicators):
            total_reward += self.feedback_weights['correction']
        
        # 3. Check for repetitive responses
        if self._is_repetitive_response(bot_response, conversation_history):
            total_reward += self.feedback_weights['repetition']
        
        # 4. Response relevance based on message similarity
        relevance_score = self._calculate_response_relevance(user_message, bot_response)
        if relevance_score > 0.7:  # High relevance threshold
            total_reward += self.feedback_weights['relevance']
        elif relevance_score < 0.3:  # Low relevance penalty
            total_reward -= self.feedback_weights['relevance']
        
        # 5. Conversation length bonus (longer conversations are generally better)
        conversation_length = len(conversation_history)
        if conversation_length > 5:
            total_reward += 0.05 * min(conversation_length - 5, 10)  # Cap bonus
        
        return total_reward
    
    def _is_repetitive_response(self, current_response: str, 
                              conversation_history: List[Dict]) -> bool:
        """Check if the current response is repetitive."""
        if len(conversation_history) < 2:
            return False
        
        # Check last few bot responses
        recent_responses = [
            turn['bot_response'] for turn in conversation_history[-3:]
        ]
        
        # Simple similarity check
        for past_response in recent_responses:
            if self._calculate_text_similarity(current_response, past_response) > 0.8:
                return True
        
        return False
    
    def _calculate_response_relevance(self, user_message: str, bot_response: str) -> float:
        """Calculate how relevant the bot response is to the user message."""
        # Simple keyword-based relevance (can be improved with better NLP)
        user_words = set(user_message.lower().split())
        bot_words = set(bot_response.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        
        user_words = user_words - stop_words
        bot_words = bot_words - stop_words
        
        if not user_words:
            return 0.5  # Neutral if no meaningful words
        
        # Calculate Jaccard similarity
        intersection = len(user_words.intersection(bot_words))
        union = len(user_words.union(bot_words))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def detect_conversation_end_signals(self, user_message: str) -> bool:
        """Detect if user is trying to end the conversation."""
        end_signals = [
            'bye', 'goodbye', 'see you', 'thanks', 'thank you',
            'that\'s all', 'nothing else', 'i\'m done', 'gotta go'
        ]
        
        return any(signal in user_message.lower() for signal in end_signals)
    
    def get_feedback_summary(self, session_id: Optional[str] = None, 
                           days: int = 7) -> Dict:
        """Get summary of feedback for analysis."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Filter feedback by date and session if specified
        relevant_feedback = [
            fb for fb in self.feedback_history
            if fb['timestamp'] >= cutoff_date
        ]
        
        if not relevant_feedback:
            return {
                'total_feedback': 0,
                'average_reward': 0.0,
                'feedback_distribution': {},
                'trend': 'neutral'
            }
        
        # Calculate statistics
        total_feedback = len(relevant_feedback)
        total_reward = sum(fb['reward'] for fb in relevant_feedback)
        average_reward = total_reward / total_feedback
        
        # Feedback type distribution
        feedback_distribution = {}
        for fb in relevant_feedback:
            fb_type = fb['type']
            if fb_type not in feedback_distribution:
                feedback_distribution[fb_type] = {'count': 0, 'avg_reward': 0.0}
            
            feedback_distribution[fb_type]['count'] += 1
            feedback_distribution[fb_type]['avg_reward'] += fb['reward']
        
        # Calculate averages
        for fb_type in feedback_distribution:
            count = feedback_distribution[fb_type]['count']
            feedback_distribution[fb_type]['avg_reward'] /= count
        
        # Determine trend (compare recent vs older feedback)
        if len(relevant_feedback) >= 10:
            recent_rewards = [fb['reward'] for fb in relevant_feedback[-5:]]
            older_rewards = [fb['reward'] for fb in relevant_feedback[-10:-5]]
            
            recent_avg = np.mean(recent_rewards)
            older_avg = np.mean(older_rewards)
            
            if recent_avg > older_avg + 0.1:
                trend = 'improving'
            elif recent_avg < older_avg - 0.1:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
        
        return {
            'total_feedback': total_feedback,
            'average_reward': average_reward,
            'feedback_distribution': feedback_distribution,
            'trend': trend,
            'recent_performance': np.mean([fb['reward'] for fb in relevant_feedback[-5:]]) if len(relevant_feedback) >= 5 else average_reward
        }
    
    def suggest_improvements(self, feedback_summary: Dict) -> List[str]:
        """Suggest improvements based on feedback analysis."""
        suggestions = []
        
        avg_reward = feedback_summary.get('average_reward', 0.0)
        trend = feedback_summary.get('trend', 'neutral')
        distribution = feedback_summary.get('feedback_distribution', {})
        
        # General performance suggestions
        if avg_reward < -0.2:
            suggestions.append("Overall performance is below average. Consider reviewing response quality.")
        elif avg_reward > 0.3:
            suggestions.append("Great performance! Keep up the good work.")
        
        # Trend-based suggestions
        if trend == 'declining':
            suggestions.append("Performance is declining. Review recent interactions for issues.")
        elif trend == 'improving':
            suggestions.append("Performance is improving! Continue current strategies.")
        
        # Specific feedback type suggestions
        if 'dislike' in distribution and distribution['dislike']['count'] > 3:
            suggestions.append("High number of dislikes. Review response relevance and quality.")
        
        if 'correction' in distribution and distribution['correction']['avg_reward'] < -0.4:
            suggestions.append("Users frequently correct responses. Improve understanding accuracy.")
        
        if 'repetition' in distribution:
            suggestions.append("Avoid repetitive responses. Increase response variety.")
        
        return suggestions if suggestions else ["Continue monitoring feedback for improvement opportunities."]