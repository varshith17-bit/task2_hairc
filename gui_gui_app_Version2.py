import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model('../saved_model/long_hair_model.h5')

def predict_image(img_path, model):
    # Load and preprocess image (adapt as per your model's needs)
    img = Image.open(img_path).resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    # Dummy logic: replace with your age/hair/gender logic
    pred_label = np.argmax(prediction, axis=1)[0]
    if pred_label == 0:
        return "Male"
    elif pred_label == 1:
        return "Female"
    else:
        return "Other"

def upload_action():
    file_path = filedialog.askopenfilename()
    result = predict_image(file_path, model)
    result_label.config(text=f"Prediction: {result}")

root = tk.Tk()
root.title("Long Hair Gender Identification")

upload_btn = tk.Button(root, text='Upload Image', command=upload_action)
upload_btn.pack()

result_label = tk.Label(root, text="Prediction: ")
result_label.pack()

root.mainloop()