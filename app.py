# problems:
# saving new chat does not seem tto work properly
# something related to session_state.session_key is going wrong

# test.py ko sabai kura run vayooooo probably because of from test import *
import os
import logging
from typing import Optional, Tuple
import time
import requests
from PIL import Image
import streamlit as st
from io import BytesIO
from config import Config
from db import db_manager
import utils
import caption_generator
from resnet import *
from transformer import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResponseGenerator:
    @staticmethod
    def stream_response(img: Optional[Image.Image], valid: bool):
        """Generate streaming response with improved error handling."""
        try:
            if valid and img:
                response = caption_generator.generate_caption(img)
                for i, word in enumerate(response):
                    yield ResponseGenerator._format_word(word, i, len(response))
                    time.sleep(0.1)
            else:
                error_msg = "Given URL is Invalid. Please Input a Valid URL."
                for word in error_msg.split():
                    yield word + " "
                    time.sleep(0.1)
        except Exception as e:
            logger.error(f"Response generation error: {str(e)}")
            yield "An error occurred while generating the caption. Please try again."

    @staticmethod
    def _format_word(word: str, index: int, total_length: int) -> str:
        """Format words in the response."""
        if index == 0:
            return word.capitalize() + " "
        elif index == total_length - 1:
            return word + "."
        return word + " "

class ImageHandler:
    @staticmethod
    def get_image(img_url: str) -> Tuple[Optional[Image.Image], Optional[str]]:
        """Get image from URL with improved error handling."""
        try:
            response = requests.get(img_url, timeout=10)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)), None
        except requests.RequestException as e:
            logger.error(f"Failed to fetch image: {str(e)}")
            return None, f"Failed to load image: {str(e)}"
        except Exception as e:
            logger.error(f"Image processing error: {str(e)}")
            return None, f"Failed to process image: {str(e)}"

class SessionManager:
    @staticmethod
    def initialize_session_state():
        """Initialize or reset session state variables."""
        if "messages" not in st.session_state:
            st.session_state.messages = []
            st.session_state.session_key = "New Session"
            st.session_state.new_session_key = None
            st.session_state.session_index_tracker = "New Session"
            st.session_state.uploader_reset_counter = 0

    @staticmethod
    def save_chat_history():
        """Save chat history with error handling."""
        if st.session_state.messages:
            session_key = (
                st.session_state.new_session_key
                if st.session_state.session_key == "New Session"
                else st.session_state.session_key
            )

            logger.debug(f"Saving chat history for session_key: {session_key}")
            logger.debug(f"Messages to save: {st.session_state.messages}")

            success = db_manager.save_chat_history(session_key, st.session_state.messages)

            if not success:
                logger.error("Failed to save chat history.")
                st.error("Failed to save chat history. Please try again.")
            else:
                logger.info("Chat history saved successfully.")
            # if st.session_state.session_key == "New Session":
            #     st.session_state.new_session_key = utils.get_timestamp()
            #     success = db_manager.save_chat_history(
            #         st.session_state.new_session_key, 
            #         st.session_state.messages
            #     )
            # else:
            #     success = db_manager.save_chat_history(
            #         st.session_state.session_key, 
            #         st.session_state.messages
            #     )
            
            # if not success:
            #     st.error("Failed to save chat history. Please try again.")

def main():
    # Configure page
    st.set_page_config(
        page_title=Config.APP_TITLE,
        page_icon='✨',
        layout="wide"
    )
    
    # Initialize session state
    SessionManager.initialize_session_state()
    
    # Setup UI
    st.title(Config.APP_TITLE)
    st.markdown(
        f'### {Config.APP_DESCRIPTION}',
        unsafe_allow_html=False
    )

    # Sidebar setup
    with st.sidebar:
        st.title("Chat Sessions")
        
        # Get cached sessions
        chat_sessions = ["New Session"] + db_manager.get_all_sessions()
        
        # Session selection
        st.selectbox(
            "Select Chat Session",
            chat_sessions,
            key="session_key",
            index=chat_sessions.index(st.session_state.session_index_tracker),
            on_change=lambda: setattr(st.session_state, 'session_index_tracker', st.session_state.session_key)
        )

        # File uploader with progress
        st.file_uploader(
            "...Or upload a Local Image",
            type=Config.ALLOWED_IMAGE_TYPES,
            key=f"file_uploader_{st.session_state.uploader_reset_counter}"
        )

    # Load or clear chat history
    if st.session_state.session_key != "New Session":
        st.session_state.messages = db_manager.load_chat_history(st.session_state.session_key)
    else:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                _display_user_message(message)
            else:
                st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Input Image URL"):
        _handle_url_input(prompt)
    elif upload_image := st.session_state.get(f"file_uploader_{st.session_state.uploader_reset_counter}"):
        _handle_file_upload(upload_image)

    # Save chat history
    # print(st.session_state.messages)
    SessionManager.save_chat_history()

def _display_user_message(message):
    """Display user message based on type."""
    if message["type"] == "url":
        img, error = ImageHandler.get_image(message["content"])
        if img:
            st.image(img, width=Config.IMAGE_DISPLAY_WIDTH)
        else:
            st.markdown(message["content"])
            if error:
                st.error(error)
    elif message["type"] == "upload":
        img, error = utils.base64_to_image(message["content"])
        if img:
            st.image(img, width=Config.IMAGE_DISPLAY_WIDTH)
        elif error:
            st.error(error)

def _handle_url_input(prompt):
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "type": "url"
    })

    img, error = ImageHandler.get_image(prompt)
    
    with st.chat_message("user"):
        if img:
            st.image(img, width=Config.IMAGE_DISPLAY_WIDTH)
        else:
            st.markdown(prompt)
            if error:
                st.error(error)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Collect the streamed response
        for chunk in ResponseGenerator.stream_response(img, img is not None):
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "type": "url"
        })

def _handle_file_upload(upload_image):
    """Handle uploaded image file from user."""
    # Convert uploaded file to base64 for storage
    base64_image = utils.image_to_base64(upload_image)
    
    st.session_state.messages.append({
        "role": "user",
        "type": "upload",
        "content": base64_image
    })

    with st.chat_message("user"):
        img = Image.open(upload_image)
        st.image(img, width=Config.IMAGE_DISPLAY_WIDTH)

    with st.chat_message("assistant"):
        response = st.write_stream(
            ResponseGenerator.stream_response(img, True)
        )

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "type": "upload"
        })

if __name__ == "__main__":
    main()