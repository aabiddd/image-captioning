import pymongo
from datetime import datetime

connection_string = "mongodb+srv://user:user123@cluster0.okftc.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = pymongo.MongoClient(connection_string)

db = client["ImageCaptioningDB"]
collection = db.get_collection('chat_history')
# print(db['chat_history'])

def save_chat_history_db(session_id, messages):
    collection.delete_many({
        "session_id": session_id
    })
    
    for message in messages:
        collection.insert_one({
            "session_id": session_id,
            "role": message['role'],
            "type": message['type'],
            "content": message['content'],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

def load_chat_history_db(session_id):
    messages = collection.find({"session_id": session_id})  # Returns a cursor which can be iterated through
    return [{
        "role": msg['role'],
        "type": msg['type'],
        "content": msg['content']
    } for msg in messages]

def get_all_sessions():
    # Use aggregate to group by unique session_id and sort them in descending order
    sessions = collection.aggregate([
        {
            "$group": {
                "_id": "$session_id"  # Group messages by session id
            }
        },
        {
            "$sort": {
                "_id": -1  # Sort in descending order, most recent items first
            }
        }
    ])

    # Extract session ids into a list
    session_list = [session["_id"] for session in sessions]
    return session_list