import os
import base64
from PIL import Image
from io import BytesIO
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    # cloud based server
    client = MongoClient(os.getenv("CONNECTION_STRING"))
    db = client["caption_db"]
    return db

def save_chat_history_mongo(session_key, chat_history):
    db = get_db_connection()
    collection = db["caption_db"]

    # Insert a new document or update existing one
    collection.update_one(
        {"session_key": session_key}, # Match document by session_key
        {"$set": {"messages": chat_history, "timestamp": get_timestamp()}}, # update or set messages
        upsert= True # Create new document if one does not exist
    )

def load_chat_history_mongo(session_key):
    db = get_db_connection()
    collection = db["caption_db"]
    document = collection.find_one({"session_key": session_key})
    if document:
        return document["messages"]
    else:
        return []
    
def get_all_sessions():
    db = get_db_connection()
    collection = db["caption_db"]

    # sort by timestamp field in descending order to get most recent sessions at t op
    sessions = collection.find({}, {"session_key": 1, "_id": 0}).sort("timestamp", -1) # Fetch only session keys
    session_list = [session["session_key"] for session in sessions]
    return session_list

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def base64_to_image(base64_str: str) -> Image.Image:
    decoded = base64.b64decode(base64_str)
    return Image.open(BytesIO(decoded))