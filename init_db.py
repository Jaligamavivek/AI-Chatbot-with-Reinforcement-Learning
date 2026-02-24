"""
Database initialization script for the chatbot system.
Creates necessary directories and initializes the database.
"""

import os
from database import DatabaseManager

def initialize_database():
    """Initialize the database and create necessary directories."""
    
    # Create data directory if it doesn't exist
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")
    
    # Initialize database
    db_path = os.path.join(data_dir, "conversations.db")
    db_manager = DatabaseManager(db_path)
    
    print(f"Database initialized at: {db_path}")
    print("Tables created:")
    print("- conversations: stores chat interactions")
    print("- feedback: stores user feedback")
    print("- q_table: stores Q-learning values")
    
    return db_manager

if __name__ == "__main__":
    print("Initializing Intelligent Chatbot Database...")
    db_manager = initialize_database()
    print("Database initialization complete!")
    
    # Test database connection
    try:
        stats = db_manager.get_feedback_stats()
        print("Database connection test: SUCCESS")
    except Exception as e:
        print(f"Database connection test: FAILED - {e}")