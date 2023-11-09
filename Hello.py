# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger
import streamlit as st
import tempfile
import os
from ultralytics import YOLO
from PIL import Image

# Load the YOLO model
model = YOLO(r'C:\naveen\cropping weights2.pt')

# Create a folder to store uploaded images
uploaded_images_folder = 'uploaded_images'
os.makedirs(uploaded_images_folder, exist_ok=True)

# Define the Streamlit app
def main():
    st.title("Object Detection and Cropping")

    # Upload multiple images
    uploaded_images = st.file_uploader("Upload multiple images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_images:
        # Create a list to store the paths of the uploaded images
        image_paths = []

        for i, uploaded_image in enumerate(uploaded_images):
            # Save the uploaded image to the uploaded_images folder with its original name
            temp_image_path = os.path.join(uploaded_images_folder, uploaded_image.name)
            with open(temp_image_path, 'wb') as temp_file:
                temp_file.write(uploaded_image.read())
            image_paths.append(temp_image_path)

            # Display the name of the uploaded image
            st.write(f"Uploaded Image {i+1}: {uploaded_image.name}")

        if st.button("Detect and Crop All Images"):
            # Process all the uploaded images and save the detection results with original names
            with st.spinner("Detecting objects and cropping all images..."):
                for i, image_path in enumerate(image_paths):
                    results = model.predict(image_path, save=True, imgsz=320, conf=0.25, save_crop=True)
                    st.success(f"Detection and cropping for Image {i+1} completed!")

if __name__ == "__main__":
    main()





