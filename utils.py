import json
import base64
from io import BytesIO
from PIL import Image
from datetime import datetime

def save_chat_history_json(chat_history, file_path):
    with open(file_path, "w") as f:
        # chat_history is a list of dict in the format: {'role': 'user/assistant', 'content': 'prompt'}
        json_data = [message for message in chat_history]
        json.dump(json_data, f)

def load_chat_history_json(file_path):
    with open(file_path, "r") as f:
        json_data = json.load(f)
        return json_data

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H_%M_%S")    

def image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def base64_to_image(base64_str: str) -> Image.Image:
    decoded = base64.b64decode(base64_str)
    return Image.open(BytesIO(decoded))