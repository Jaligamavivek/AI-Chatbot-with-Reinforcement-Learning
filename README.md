# AI Chatbot with Reinforcement Learning

A smart chatbot that learns from user feedback using Q-Learning (Reinforcement Learning).

## 🎯 What is This?

This chatbot gets smarter over time by learning from your feedback:
- You chat with the bot
- You give feedback (👍 like, 👎 dislike, or ⭐ rating)
- The bot remembers what responses you liked
- The bot improves its answers automatically

## 🧠 How Reinforcement Learning Works

```
1. User asks: "Hello"
2. Bot responds: "Hi there!"
3. User clicks: 👍 (like)
4. Bot learns: "This response was good, use it more!"
5. Next time: Bot prefers responses that got positive feedback
```

## 📦 Installation

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Initialize Database
```bash
python init_db.py
```

### 3. Create Demo User (Optional)
```bash
python create_demo_user.py
```

## 🚀 Running the Application

```bash
python app.py
```

Then open your browser to: **http://localhost:5000**

## 🔐 Login

**Demo Account:**
- Username: `demo`
- Password: `demo123`

Or create your own account using the signup page.

## 💬 How to Use

1. **Login** to the application
2. **Type a message** in the chat box
3. **Get a response** from the AI
4. **Give feedback:**
   - Click 👍 if you like the response
   - Click 👎 if you don't like it
   - Click ⭐ to rate 1-5 stars
5. **Watch it learn!** The bot improves with each feedback

## 📊 Features

- ✅ **Q-Learning Algorithm** - Real reinforcement learning
- ✅ **User Feedback System** - Like/dislike/rating buttons
- ✅ **Learning Analytics** - See how the bot improves
- ✅ **User Authentication** - Secure login/signup
- ✅ **Conversation History** - Saves all chats
- ✅ **Real-time Learning** - Bot updates immediately

## 🛠️ Technology Stack

**Backend:**
- Python 3.7+
- Flask (Web framework)
- SQLite (Database)
- NumPy (Q-Learning calculations)

**Frontend:**
- HTML5, CSS3, JavaScript
- Font Awesome (Icons)
- Responsive design

**AI/ML:**
- Q-Learning (Reinforcement Learning)
- NLP (Natural Language Processing)
- TF-IDF (Text analysis)

## 📁 Project Structure

```
ai-chatbot/
├── app.py                 # Main Flask application
├── chatbot.py            # Chatbot logic
├── rl_agent.py           # Q-Learning implementation
├── nlp_processor.py      # Text processing
├── feedback_handler.py   # User feedback processing
├── database.py           # Database operations
├── auth.py               # User authentication
├── init_db.py           # Database setup
├── create_demo_user.py  # Demo user creation
├── requirements.txt      # Python dependencies
├── templates/           # HTML templates
│   ├── login.html
│   ├── signup.html
│   └── dashboard.html
├── static/              # CSS files
│   ├── style.css
│   └── auth.css
└── data/               # Database files
    ├── conversations.db
    └── users.db
```

## 🎓 Understanding the Code

### Q-Learning (rl_agent.py)
The bot uses a Q-table to store scores for each response:
- **State**: User's question/intent
- **Action**: Bot's response
- **Reward**: User feedback (+1 for like, -1 for dislike)
- **Learning**: Updates Q-values to prefer better responses

### Feedback System (feedback_handler.py)
Converts user feedback into rewards:
- 👍 Like = +1.0 reward
- 👎 Dislike = -1.0 reward
- ⭐⭐⭐⭐⭐ (5 stars) = +1.0 reward
- ⭐ (1 star) = -1.0 reward

### NLP Processing (nlp_processor.py)
Understands user messages:
- Detects intent (greeting, question, help, etc.)
- Extracts context from conversation
- Generates appropriate responses

## 📈 Monitoring Learning

The dashboard shows real-time statistics:
- **Total Episodes**: How many times the bot learned
- **Q-Table Size**: Number of learned state-action pairs
- **Average Reward**: How well the bot is performing
- **Feedback Count**: Total user feedback received

## 🔧 Configuration

Edit these files to customize:
- `rl_agent.py` - Learning rate, discount factor
- `nlp_processor.py` - Response templates, intents
- `app.py` - Server settings, secret key

## 🐛 Troubleshooting

**Problem: Database error**
```bash
# Solution: Reinitialize database
python init_db.py
```

**Problem: Login not working**
```bash
# Solution: Create demo user
python create_demo_user.py
```

**Problem: Port already in use**
```bash
# Solution: Change port in app.py (last line)
app.run(debug=True, host='0.0.0.0', port=5001)
```

## 📝 Requirements

- Python 3.7 or higher
- 2GB RAM minimum
- Modern web browser
- Internet (for CDN resources like Font Awesome)

## 🎯 Learning Goals

This project demonstrates:
1. **Reinforcement Learning** - Q-Learning algorithm
2. **Web Development** - Full-stack application
3. **Database Management** - SQLite operations
4. **User Authentication** - Secure login system
5. **Real-time Updates** - Dynamic UI changes
6. **NLP Basics** - Text processing and intent detection

## 🚀 Future Enhancements

- [ ] Deep Q-Learning with neural networks
- [ ] Multi-language support
- [ ] Voice input/output
- [ ] Advanced NLP with transformers
- [ ] User personalization
- [ ] Export learning data

## 📄 License

This project is for educational purposes.

## 🤝 Contributing

Feel free to fork, modify, and improve this project!

---

**Made with ❤️ using Python and Reinforcement Learning**
