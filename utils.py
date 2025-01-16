import base64
import logging
from io import BytesIO
from datetime import datetime
from typing import Optional, Tuple
from PIL import Image
from functools import lru_cache

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_timestamp() -> str:
    """
    Generate current timestamp in consistent format.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def validate_image(image: Image.Image, max_size: int) -> Tuple[bool, Optional[str]]:
    """
    Validate image size and format.
    Returns:
    (is_valid, error_message)
    """
    try:
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format="PNG")
        size = img_byte_arr.tell()

        if size > max_size:
            return False, f"Image Size {size/1024/1024:.1f}MB exceeds maximum allowed size {max_size/1024/1024}MB"
        
        return True, None
    
    except Exception as e:
        logger.error(f"Image Validation Error: {str(e)}")
        return False, "Invalid Image Format"

def image_to_base64(uploadFile, max_size: int = 5*1024*1024) -> Tuple[Optional[str], Optional[str]]:
    """
    Convert PIL Image to base64 string with validation.
    Returns (base64_string, error_message)
    """
    try:
        image = Image.open(uploadFile)
        # Validate image
        is_valid, error = validate_image(image, max_size)
        if not is_valid:
            return None, error
            
        # Convert to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG", optimize=True)
        return base64.b64encode(buffered.getvalue()).decode("utf-8"), None
        
    except Exception as e:
        logger.error(f"Image to base64 conversion error: {str(e)}")
        return None, "Failed to process image"

def base64_to_image(base64_str: str) -> Tuple[Optional[Image.Image], Optional[str]]:
    """
    Convert base64 string to PIL Image.
    Returns (image, error_message)
    """
    try:
        decoded = base64.b64decode(base64_str)
        return Image.open(BytesIO(decoded)), None
    except Exception as e:
        logger.error(f"Base64 to image conversion error: {str(e)}")
        return None, "Failed to decode image"
    
@lru_cache(maxsize=1000)
def cached_timestamp(identifier: str) -> str:
    """
    Cached version of timestamp generation for repeated operations.
    """
    return get_timestamp()

def cleanup_old_sessions(sessions: list, max_sessions: int) -> list:
    """
    Remove oldest sessions when limit is reached.
    """
    if len(sessions) > max_sessions:
        return sorted(sessions, reverse=True)[:max_sessions]
    return sessions