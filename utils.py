import json
import base64
import sqlite3
from io import BytesIO
from PIL import Image
from datetime import datetime

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")    

def image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def base64_to_image(base64_str: str) -> Image.Image:
    decoded = base64.b64decode(base64_str)
    return Image.open(BytesIO(decoded))

def save_chat_history_to_db(session_id, messages):
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()

    for message in messages:
        cursor.execute('''
            INSERT INTO chat_history (session_id, role, content, message_type, timestamp)
            VALUES(?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (session_id, message['role'], message['content'], message['type']))

    conn.commit()
    conn.close()

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

def get_session_list():
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()

    cursor.execute('''
        SELECT DISTINCT session_id FROM chat_history
    ''')

    sessions = cursor.fetchall()

    # convert the list of tuples to a simple list of sessions IDs
    session_list = [session[0] for session in sessions]

    conn.close()
    return session_list