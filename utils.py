import base64
from PIL import Image
from io import BytesIO
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