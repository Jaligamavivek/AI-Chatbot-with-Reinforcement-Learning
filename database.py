"""
Database operations for the chatbot system.
Handles conversation storage, feedback logging, and Q-table persistence.
"""

import sqlite3
import json
import pickle
from datetime import datetime
from typing import List, Dict, Any, Optional

class DatabaseManager:
    def __init__(self, db_path: str = "data/conversations.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_message TEXT NOT NULL,
                bot_response TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                context TEXT,
                intent TEXT
            )
        ''')
        
        # Feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                feedback_type TEXT NOT NULL,
                feedback_value REAL NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations (id)
            )
        ''')
        
        # Q-table storage
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS q_table (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                state_action_key TEXT UNIQUE NOT NULL,
                q_value REAL NOT NULL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_conversation(self, session_id: str, user_message: str, 
                         bot_response: str, context: str = "", 
                         intent: str = "") -> int:
        """Save a conversation turn to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO conversations (session_id, user_message, bot_response, context, intent)
            VALUES (?, ?, ?, ?, ?)
        ''', (session_id, user_message, bot_response, context, intent))
        
        conversation_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return conversation_id
    
    def save_feedback(self, conversation_id: int, feedback_type: str, 
                     feedback_value: float):
        """Save user feedback to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO feedback (conversation_id, feedback_type, feedback_value)
            VALUES (?, ?, ?)
        ''', (conversation_id, feedback_type, feedback_value))
        
        conn.commit()
        conn.close()
    
    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Get recent conversation history for a session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_message, bot_response, timestamp, context, intent
            FROM conversations 
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (session_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                'user_message': row[0],
                'bot_response': row[1],
                'timestamp': row[2],
                'context': row[3],
                'intent': row[4]
            }
            for row in reversed(rows)
        ]
    
    def save_q_value(self, state_action_key: str, q_value: float):
        """Save or update Q-value in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO q_table (state_action_key, q_value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        ''', (state_action_key, q_value))
        
        conn.commit()
        conn.close()
    
    def load_q_table(self) -> Dict[str, float]:
        """Load Q-table from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT state_action_key, q_value FROM q_table')
        rows = cursor.fetchall()
        conn.close()
        
        return {row[0]: row[1] for row in rows}
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get feedback statistics for evaluation."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                feedback_type,
                AVG(feedback_value) as avg_feedback,
                COUNT(*) as count
            FROM feedback
            GROUP BY feedback_type
        ''')
        
        stats = {}
        for row in cursor.fetchall():
            stats[row[0]] = {
                'average': row[1],
                'count': row[2]
            }
        
        conn.close()
        return stats