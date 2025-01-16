import logging
from typing import  List, Dict, Optional
from datetime import datetime, timedelta
import pymongo
from functools import lru_cache
from config import Config

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.client = None
        self.db = None
        self.collection = None
        self.connect()

    def connect(self):
        """
        Establish database connection iwth retry logic
        """
        try:
            self.client = pymongo.MongoClient(Config.MONGODB_URI)
            self.db = self.client[Config.DB_NAME]
            self.collection = self.db[Config.COLLECTION_NAME]
            self.client.server_info()  # Test Connection
            logger.info("Connected to MongoDB Atlas ✅")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    def cleanup_old_sessions(self, hours: int = 24):
        """
        Remove sessions older than specified hours.
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            result = self.collection.delete_many({
                "timestamp": {"$lt": cutoff_time.strftime("%Y-%m-%d %H:%M:%S")}
            })
            logger.info(f"Cleaned UP {result.deleted_count} old sessions")
        except Exception as e:
            logger.error(f"Session cleanup failed: {e}")

    def save_chat_history(self, session_id: str, messages: List[Dict]) -> bool:
        """
        Save chat history with error handling and validation
        """
        try:
            # Delete existing messages for session
            self.collection.delete_many({"session_id": session_id})

            # Insert new messages
            if messages:
                documents = [{
                    "session_id": session_id,
                    "role": msg['role'],
                    "type": msg['type'],
                    "content": msg['content'],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                } for msg in messages]
                self.collection.insert_many(documents)
            return True
        except Exception as e:
            logger.error(f"Failed to save chat history: {e}")
            return False
    
    def load_chat_history(self, session_id: str) -> List[Dict]:
        """
        Load chat history
        """
        try:
            messages = self.collection.find({"session_id": session_id})
            return [{
                "role": msg['role'],
                "type": msg['type'],
                "content": msg['content']
            } for msg in messages]
        except Exception as e:
            logger.error(f"Failed to load chat history: {str(e)}")
            return []
        
    @lru_cache(maxsize=1) 
    def get_all_sessions(self) -> List[str]:
        """
        Get all sessions with caching.
        """
        try:
            sessions = self.collection.aggregate([
                {"$group": {"_id": "$session_id"}},
                {"$sort": {"_id": -1}}
            ])
            return [session["_id"] for session in sessions]
        except Exception as e:
            logger.error(f"Failed to get sessions: {e}")
            return []
        

# Initialize database manager
db_manager = DatabaseManager()