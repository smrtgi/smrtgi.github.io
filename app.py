from flask import Flask, render_template, request, send_file
import requests
import io
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import os

from ultralytics import YOLO

# YOLO Saved Model
cat_model_url = 'https://github.com/fvazqu/Cat-vs-Dog/raw/main/best.pt'
local_model_path = 'best.pt'
response = requests.get(cat_model_url)

with open(local_model_path, 'wb') as file:
    file.write(response.content)

model = YOLO(local_model_path)


def process_image(image_url):
    try:
        # Load image from internet
        response = requests.get(image_url)

        # Check if the response status code indicates a successful request (e.g., 200 OK)
        if response.status_code != 200:
            raise Exception("Failed to retrieve the image. Check the URL and try again.")

        image = Image.open(BytesIO(response.content))
        image = np.asarray(image)

        # Run image on YOLO Model
        results = model.predict(image)

        # Save Image
        for r in results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            image_path = 'static/results.jpg'
            im.save(image_path)  # save image

        # Output response
        output = "The image has been processed through the YOLO model"

        return image_path, output

    except Exception as e:
        # Handle any exceptions that occurred during image retrieval or processing
        error_message = str(e)
        return None, error_message  # Return None for the image and the error message

# ChatGPT Part
import openai

openai.api_key = "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"
messages = [{"role": "system", "content": "You are a intelligent assistant."}]

def chat_with_gpt(input_text):
    message = input_text
    if message:
      messages.append(
          {"role": "user", "content": message},
      )
      chat = openai.ChatCompletion.create(
          model="gpt-3.5-turbo", messages=messages
      )

    reply = chat.choices[0].message.content

    # print(f"ChatGPT: {reply}")
    chat_response = f"ChatGPT: {reply}"
    messages.append({"role": "assistant", "content": reply})
    print(chat_response)
    return chat_response

app = Flask(__name__)


@app.route("/")
def hello_world():
    return render_template('index.html')


@app.route('/img', methods=["GET", "POST"])
def predict_img():
    if request.method == 'POST':
        # Get the URL from the form
        url = request.form['url']

        # Set the image_url variable for displaying in the template
        image_url = url

        # Process the image using YOLO (replace with your YOLO code)
        image_results, output = process_image(image_url)
        print(image_results, output)

    return render_template('index.html', results=image_results)


@app.route('/chat', methods=["POST"])
def chatgpt():
    if request.method == 'POST':
        chat = request.form['chat']
        chat_response = chat_with_gpt(chat)
        return render_template('index.html', reply=chat_response, user=chat)


if __name__ == '__main__':
    app.run(debug=True)
