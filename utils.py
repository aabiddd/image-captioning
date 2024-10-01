import json
import base64
import sqlite3
from io import BytesIO
from PIL import Image
from datetime import datetime
import os

# Function to get current timestamp
def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Function to convert image to base64 string
def image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Function to convert base64 string back to image
def base64_to_image(base64_str: str) -> Image.Image:
    decoded = base64.b64decode(base64_str)
    return Image.open(BytesIO(decoded))

# Function to save chat history to the database
def save_chat_history_to_db(session_id, messages):
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()

    # Create table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            message_type TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    for message in messages:
        cursor.execute('''
            INSERT INTO chat_history (session_id, role, content, message_type)
            VALUES(?, ?, ?, ?)
        ''', (session_id, message['role'], message['content'], message['type']))

    conn.commit()
    conn.close()

# Function to load chat history from the database
def load_chat_history_from_db(session_id):
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()

    cursor.execute('''
        SELECT role, content, message_type FROM chat_history
        WHERE session_id = ?
        ORDER BY timestamp ASC
    ''', (session_id, ))

    rows = cursor.fetchall()

    messages = [{'role': row[0], 'content': row[1], 'type': row[2]} for row in rows]

    conn.close()
    return messages

# Function to get a list of distinct session IDs from the database
def get_session_list():
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()

    cursor.execute('''
        SELECT DISTINCT session_id FROM chat_history
    ''')
    
    sessions = cursor.fetchall()

    # Convert the list of tuples to a simple list of session IDs
    session_list = [session[0] for session in sessions]

    conn.close()
    return session_list

# Function to save chat history as a JSON file
def save_chat_history_json(messages, file_path):
    try:
        with open(file_path, 'w') as json_file:
            json.dump(messages, json_file, indent=4)
    except Exception as e:
        print(f"Error saving chat history to JSON: {e}")
