import sqlite3 

# initialie database connection
def init_db():
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()

    # table to store chat history
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS chat_history (
        session_id TEXT,
        role TEXT,
        content TEXT,
        message_type TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    conn.commit()
    conn.close()

# call this fxn directly when running this script
if __name__ == "__main__":
    init_db()
    print("Database innitialized!")