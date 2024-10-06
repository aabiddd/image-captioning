# streamlit run app.py --server.enableXsrfProtection false
import os
import time
import utils
import requests
from PIL import Image
import streamlit as st
from io import BytesIO

# Streamed response emulator
def response_generator(prompt: str, validness: bool):
    # if given prompt is a valid URL
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

# get image via requests
def get_image(img_url: str):
    img = None        
    try:
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content))
    except Exception as e:
        st.error(f"Failed to load image from URL. \nError: {e}")
    return img

# save chat history based on current session key
def save_chat_history():
    # Check if there are any messages to save
    if st.session_state.messages != []:
        if st.session_state.session_key == "New Session":
            # generate new session key based on current time
            st.session_state.new_session_key = utils.get_timestamp()
            utils.save_chat_history_mongo(st.session_state.new_session_key, st.session_state.messages)
        else:
            utils.save_chat_history_mongo(st.session_state.session_key, st.session_state.messages)
        
def track_index():
    st.session_state.session_index_tracker = st.session_state.session_key

def main():
    # title
    st.title("ICRT")
    st.markdown('''
        ### An Integrative Approach To Image Captioning with ResNet and Transformers
    ''')

    # chat  session sidebar
    st.sidebar.title("Chat Sessions")

    #  List of sessions that will be displayed in the selectbox
    chat_sessions = ["New Session"] + utils.get_all_sessions()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [] # empty history
        st.session_state.session_key = "New Session"
        st.session_state.new_session_key = None
        st.session_state.session_index_tracker = "New Session"

    # Initialize a counter in the session state if it doesnt' exist
    if 'uploader_reset_counter' not in st.session_state:
        st.session_state.uploader_reset_counter = 0

    # save the contents of new_session based on the new_session_key
    if st.session_state.session_key == "New Session" and st.session_state.new_session_key != None:
        st.session_state.session_index_tracker = st.session_state.new_session_key
        st.session_state.new_session_key = None

    # track the index of current session
    index = chat_sessions.index(st.session_state.session_index_tracker)

    # display chat sessions inside the sidebar
    with st.sidebar:
        selectbox = st.selectbox("Select Chat Session", chat_sessions, key="session_key", index=index, on_change=track_index)

    # load chat history when old session is selected
    if st.session_state.session_key != "New Session":
        st.session_state.messages = utils.load_chat_history_mongo(st.session_state.session_key)
    else:
        st.session_state.messages = []


    # Padding
    for _ in range(15):
        st.sidebar.text("")

    # Generate a unique key for the file uploader widget using the counter
    unique_uploader_key = f"file_uploader_{st.session_state.uploader_reset_counter}"

    # upload local image via file uploader widget with a unique key
    upload_image = st.sidebar.file_uploader("...Or upload a Local Image", type=["png", "jpg", "jpeg"], key=unique_uploader_key)

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # user message (images)
            if message["role"] == "user":
                # url-based
                if message["type"] == "url":
                    img = get_image(message["content"])
                    if img:
                        st.image(img, width=400)
                    else:
                        # if invalid URL, display the url user entered
                        st.markdown(message["content"])
                
                # upload-based
                elif message["type"] == "upload":
                    img = utils.base64_to_image(message["content"])
                    st.image(img, width=400)
            
            # display bot responses
            else:
                st.markdown(message["content"])

    # Accept user input
    if (prompt := st.chat_input("Input Image URL")) or upload_image:
        # if prompt was given
        if prompt:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt, "type":"url"})

            # get image based on user's input url
            img = get_image(prompt)

            # when valid image-url is given, img will not be None, else None
            if img:
                # Display user input image in chat message container
                with st.chat_message("user"):
                    st.image(img, width=400)

                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    response = st.write_stream(response_generator(prompt, True))
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response, "type":"url"})

            else:
                # Display user message in chat message container
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    response = st.write_stream(response_generator(prompt, False))
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response, "type": "url"})

        # else if, image was "uploaded"
        elif upload_image:
            img = Image.open(upload_image)
            # decode the image into string, so that it becomes JSON serializable
            img_base64 = utils.image_to_base64(img)

            with st.chat_message("user"):
                st.image(img, width=400)
                st.session_state.messages.append({"role": "user", "content": img_base64, "type": "upload"})

            with st.chat_message("assistant"):
                response = st.write_stream(response_generator(prompt, True))
                st.session_state.messages.append({"role": "assistant", "content": response, "type": "upload"})

            # the image date gets saved into this variable, due to which the image gets shown on any other sessions (including new session)
            # Increment the counter to change the key, effectively resetting the file uploader
            st.session_state.uploader_reset_counter += 1

    # save chat history before each app rerun
    save_chat_history()

if __name__ == "__main__":
    main()