import os
import time
import utils
import requests
from PIL import Image
import streamlit as st
from io import BytesIO

def response_generator(prompt: str, validness: bool):
    if validness:
        response = "This is some caption, that will get replaced by the model."
        for word in response.split():
            yield word + " "
            time.sleep(0.1)
    else:
        response = "Given URL is Invalid. Please Input a Valid URL."
        for word in response.split():
            yield word + " "
            time.sleep(0.1)

def get_image(img_url: str):
    img = None        
    try:
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content))
    except Exception as e:
        st.error(f"Failed to load image from URL. \nError: {e}")
    return img

def save_chat_history():
    if st.session_state.messages:
        session_key = st.session_state.session_key
        utils.save_chat_history_to_db(session_key, st.session_state.messages)

def track_index():
    st.session_state.session_index_tracker = st.session_state.session_key

def main():
    st.title("ICRT")
    st.markdown('''### An Integrative Approach To Image Captioning with ResNet and Transformers''')

    st.sidebar.title("Chat Sessions")
    chat_sessions = ["New Session"] + utils.get_session_list()

    # used a temporary variable, ps only update session key if it differs from the current value 
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.session_key = "New Session"
        st.session_state.session_index_tracker = "New Session"

    if 'uploader_reset_counter' not in st.session_state:
        st.session_state.uploader_reset_counter = 0

    # Determine the index of the selected session in the sidebar
    index = chat_sessions.index(st.session_state.session_index_tracker)

    # Selectbox for choosing the session
    with st.sidebar:
        selected_session = st.selectbox("Select Chat Session", chat_sessions, index=index, on_change=track_index)

    # Update session state based on selection in the sidebar
    if selected_session != st.session_state.session_key:
        st.session_state.session_key = selected_session
        if selected_session == "New Session":
            st.session_state.messages = []
        else:
            st.session_state.messages = utils.load_chat_history_from_db(selected_session)

    for _ in range(15):
        st.sidebar.text("")

    unique_uploader_key = f"file_uploader_{st.session_state.uploader_reset_counter}"
    upload_image = st.sidebar.file_uploader("...Or upload a Local Image", type=["png", "jpg", "jpeg"], key=unique_uploader_key)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                if message["type"] == "url":
                    img = get_image(message["content"])
                    if img:
                        st.image(img, width=400)
                    else:
                        st.markdown(message["content"])
                elif message["type"] == "upload":
                    img = utils.base64_to_image(message["content"])
                    st.image(img, width=400)
            else:
                st.markdown(message["content"])

    # Handle new inputs
    if (prompt := st.chat_input("Input Image URL")) or upload_image:
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt, "type": "url"})
            img = get_image(prompt)

            if img:
                with st.chat_message("user"):
                    st.image(img, width=400)

                with st.chat_message("assistant"):
                    response = st.write_stream(response_generator(prompt, True))
                st.session_state.messages.append({"role": "assistant", "content": response, "type": "url"})
            else:
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    response = st.write_stream(response_generator(prompt, False))
                st.session_state.messages.append({"role": "assistant", "content": response, "type": "url"})
        elif upload_image:
            img = Image.open(upload_image)
            img_base64 = utils.image_to_base64(img)

            with st.chat_message("user"):
                st.image(img, width=400)
                st.session_state.messages.append({"role": "user", "content": img_base64, "type": "upload"})

            with st.chat_message("assistant"):
                response = st.write_stream(response_generator("", True))
                st.session_state.messages.append({"role": "assistant", "content": response, "type": "upload"})

            st.session_state.uploader_reset_counter += 1

    # If it's a new session, generate a unique session key
    if st.session_state.session_key == "New Session" and st.session_state.messages:
        new_session_key = utils.get_timestamp()
        st.session_state.session_key = new_session_key
        st.session_state.session_index_tracker = new_session_key

    save_chat_history()

if __name__ == "__main__":
    main()
