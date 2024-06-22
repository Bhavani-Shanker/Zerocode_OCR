import streamlit as st
from paddleocr import PaddleOCR
import numpy as np
import cv2
from PIL import Image

# Initialize PaddleOCR with English model (default)
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Streamlit title
st.title("OCR Text Extraction - Zerocode Innovations")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    image = Image.open(uploaded_file)
    img = np.array(image)

    # Perform OCR on the image
    result = ocr.ocr(np.array(image), cls=True)

    # Extract the text word by word and store bounding boxes
    word_by_word_text = []
    word_boxes = []
    for line in result:
        for word_info in line:
            word_text = word_info[1][0]
            word_by_word_text.append(word_text)
            word_boxes.append(word_info[0])  # Store bounding boxes for each word

    # Extract the text line by line
    line_by_line_text = []
    for line in result:
        line_text = " ".join([word_info[1][0] for word_info in line])
        line_by_line_text.append(line_text)

    # Create two columns
    col1, col2 = st.columns(2)

    # Display the image in the first column
    with col1:
        selected_word = st.selectbox("Select a word to highlight:", [""] + word_by_word_text)

        if selected_word:
            # Find the index of the selected word
            selected_word_index = word_by_word_text.index(selected_word)

            # Get the bounding box for the selected word
            selected_word_box = word_boxes[selected_word_index]

            # Draw a rectangle on the image for the selected word
            img_copy = img.copy()
            box = np.array(selected_word_box).astype(int)
            cv2.polylines(img_copy, [box], isClosed=True, color=(255, 0, 0), thickness=2)

            # Display the image with highlighted word
            st.image(img_copy, caption='Highlighted Word Region', use_column_width=True)
        else:
            st.image(image, caption='Uploaded Image', use_column_width=True)

    # Display the extracted text in the second column
    with col2:
        st.header("Extracted Text - Word by Word")
        for word_text in word_by_word_text:
            st.write(word_text)

        st.header("Extracted Text - Line by Line")
        for line_text in line_by_line_text:
            st.write(line_text)

    # Provide a download button for word by word text
    st.download_button(
        label="Download Word by Word Text",
        data="\n".join(word_by_word_text),
        file_name="word_by_word_text.txt",
        mime="text/plain"
    )

    # Provide a download button for line by line text
    st.download_button(
        label="Download Line by Line Text",
        data="\n".join(line_by_line_text),
        file_name="line_by_line_text.txt",
        mime="text/plain"
    )
