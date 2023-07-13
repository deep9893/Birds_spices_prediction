import tkinter.filedialog as filedialog
from tkinter import *
import tkinter as tk
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np

model = load_model('mobilenetv2.h5')

def prediction(path):
    img = image.load_img(path, target_size=(224, 224))
    i = image.img_to_array(img)
    i = np.expand_dims(i, axis=0)
    img = preprocess_input(i)
    pred = np.argmax(model.predict(img), axis=1)
    return pred

window = Tk()
window.geometry("600x300")
window.title("Bird Species Prediction")
l = tk.Label(text='Bird Species Prediction', font=('Arial', 18, 'bold'), bg='orange', fg='white')
l.place(x=70, y=10)

def upload_image():
    path = filedialog.askopenfilename()
    pred = prediction(path)
    result = ""
    if pred == 0:
        info="This birds mostly found Himalayas in South Asia"
        result = "ABBOTTS BABBLER\n" + info
    elif pred == 1:
        info="This birds mostly found Eastern Russia and winters in East Asia" 
        result = "BAIKAL TEAL\n" + info
    elif pred == 2:
        info="This birds mostly found in Northeastern Brazil."
        result = "CAATINGA CACHOLOTE\n" + info
    elif pred == 3:
        info="This birds mostly found in Bhutan, India, Myanmar, Nepal and Tibet."
        result = "DARJEELING WOODPECKER\n" + info
    elif pred == 4:
        info="This birds mostly found in East of the Rockies, southern Canada"
        result = "EASTERN BLUEBIRD\n" + info
    elif pred == 5:
        info="This birds mostly found in southern Asia and the Philippines."
        result = "FAIRY BLUEBIRD\n" + info
    elif pred == 6:
        info="This birds mostly found in desert regions of Arizona, California, Colorado."
        result = "GAMBELS QUAIL\n" + info
    elif pred == 7:
        info="This birds mostly found in Africa, Madagascar and Arabia"
        result = "HAMERKOP\n" + info
    elif pred == 8:
        info="This birds mostly found in  Spain and Portugal"
        result = "IBERIAN MAGPIE\n" + info
    elif pred == 9:
        info="This birds mostly found in  Americas from Mexico to Argentina,"
        result = "JABIRU\n" + info
    elif pred == 10:
        info="This birds mostly found in New Caledonia"
        result = "KAGU\n" + info
    elif pred == 11:
        info="This birds mostly found in central and western North America."
        result = "LARK BUNTING\n" + info
    elif pred == 12:
        info="This birds mostly found in northern Australia and southern New Guinea"
        result = "MAGPIE GOOSE\n" + info 
    elif pred == 13:
        info="This birds mostly found in Andaman and Nicobar Islands,"
        result = "NICOBAR PIGEON\n" + info
    elif pred == 14:
        info="This birds mostly found in Mexico"
        result = "OCELLATED TURKEY\n" + info
    print(result)
    label.config(text=result)

button = tk.Button(window, text="Upload Image", font=('Arial', 14, 'bold'), bg='orange', fg='white', command=upload_image)
button.place(x=135, y=100)

label = tk.Label(window)
label.place(x=155, y=180)

window.config(bg='yellow')
window.mainloop()
