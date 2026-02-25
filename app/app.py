"""
Flask web application for the Intelligent Chatbot with RL Feedback.
Provides web interface for user interaction and feedback collection.
"""

from flask import Flask, render_template, request, jsonify, session
import os
from datetime import datetime

from chatbot import IntelligentChatbot

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production-ai-assistant-2024'

# Initialize chatbot
chatbot = IntelligentChatbot()

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

# Main Application Routes
@app.route('/')
def index():
    """Main page - Direct access to chatbot without authentication."""
    # Create a guest session
    if 'chatbot_session_id' not in session:
        session['chatbot_session_id'] = chatbot.create_session()
    
    return render_template('dashboard.html', user={'username': 'Guest', 'user_id': 1})

@app.route('/dashboard')
def dashboard():
    """Main chat interface - Direct access."""
    # Create a guest session
    if 'chatbot_session_id' not in session:
        session['chatbot_session_id'] = chatbot.create_session()
    
    return render_template('dashboard.html', user={'username': 'Guest', 'user_id': 1})

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages."""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        # Get or create session
        if 'chatbot_session_id' not in session:
            session['chatbot_session_id'] = chatbot.create_session()
        
        session_id = session['chatbot_session_id']
        
        # Process message
        response_data = chatbot.process_message(session_id, user_message)
        
        return jsonify({
            'success': True,
            'response': response_data['response'],
            'conversation_id': response_data['conversation_id'],
            'intent': response_data['intent'],
            'confidence': response_data['confidence'],
            'metadata': response_data['metadata']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    """Handle user feedback."""
    try:
        data = request.get_json()
        conversation_id = data.get('conversation_id')
        feedback_type = data.get('feedback_type')  # 'like', 'dislike', 'rating'
        feedback_value = data.get('feedback_value')  # For ratings
        
        print(f"DEBUG: Received feedback - ID: {conversation_id}, Type: {feedback_type}, Value: {feedback_value}")
        
        if not conversation_id or not feedback_type:
            print("DEBUG: Missing required fields")
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400
        
        # Process feedback
        try:
            result = chatbot.process_explicit_feedback(
                conversation_id, feedback_type, feedback_value
            )
            print(f"DEBUG: Feedback processed successfully: {result}")
            
            return jsonify({
                'success': True,
                'message': result['message'],
                'reward': result['reward'],
                'stats': result['q_stats']
            })
        except Exception as feedback_error:
            print(f"DEBUG: Feedback processing error: {feedback_error}")
            return jsonify({'success': False, 'error': f'Feedback processing failed: {str(feedback_error)}'}), 500
        
    except Exception as e:
        print(f"DEBUG: General feedback error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/stats')
def stats():
    """Get learning and performance statistics."""
    try:
        learning_stats = chatbot.get_learning_stats()
        
        # Get session stats if available
        session_stats = {}
        if 'chatbot_session_id' in session:
            session_stats = chatbot.get_session_stats(session['chatbot_session_id'])
        
        return jsonify({
            'success': True,
            'learning_stats': learning_stats,
            'session_stats': session_stats
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reset_session', methods=['POST'])
def reset_session():
    """Reset current chat session."""
    try:
        session.pop('chatbot_session_id', None)
        new_session_id = chatbot.create_session()
        session['chatbot_session_id'] = new_session_id
        
        return jsonify({
            'success': True,
            'message': 'Session reset successfully',
            'new_session_id': new_session_id
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/export_data')
def export_data():
    """Export learning data for analysis."""
    try:
        learning_data = chatbot.export_learning_data()
        
        return jsonify({
            'success': True,
            'data': learning_data,
            'exported_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/best_responses/<intent>')
def best_responses(intent):
    """Get best learned responses for an intent."""
    try:
        responses = chatbot.get_best_responses_for_intent(intent)
        
        return jsonify({
            'success': True,
            'intent': intent,
            'best_responses': [
                {'response': resp, 'score': score}
                for resp, score in responses
            ]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'chatbot_ready': True
    })

# Removed app.run() for serverless deployment