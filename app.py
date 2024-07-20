import time
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
        st.error(f"Failed to load image from URL. Error: {e}")
    return img


def main():
    # title
    st.title("ICRT")
    st.markdown('''
        ### An Integrative Approach To Image Captioning with ResNet and Transformers
    ''')

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                img = get_image(message["content"])
                if img:
                    st.image(img, width=400)
                else:
                    st.markdown(message["content"])
            else:
                st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Input Image URL"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # get image based on user's input url
        img = get_image(prompt)

        # if valid image
        if img:
            # Display user input image in chat message container
            with st.chat_message("user"):
                st.image(img, width=400)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                response = st.write_stream(response_generator(prompt, True))
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

        else:
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                response = st.write_stream(response_generator(prompt, False))
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()