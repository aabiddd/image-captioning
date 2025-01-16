from pathlib import Path

class Config:
    # Application settings
    APP_TITLE = "ICRT"
    APP_DESCRIPTION = "An Integrative Approach To Image Captioning with ResNet and Transformers"

    # Database Settings
    MONGODB_URI = "mongodb+srv://user:user123@cluster0.okftc.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    DB_NAME = "ImageCaptioningDB"
    COLLECTION_NAME = "chat_history"

    # Image Settings
    ALLOWED_IMAGE_TYPES = ["png", "jpg", "jpeg"]
    MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB
    IMAGE_DISPLAY_WIDTH = 300

    # Session Settings
    SESSION_TIMEOUT = 3600  # 1hr
    MAX_SESSIONS_PER_USER = 50

    # Cache Settings
    # CACHE_TIMEOUT = 300  # 5minutes

     # Model settings
    MODEL_PATH = Path("./model/checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar")
    WORD_MAP_PATH = Path("./wordEmbeddings/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json")