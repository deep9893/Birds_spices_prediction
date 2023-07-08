#load the model
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

path = r"datasets/test/FAIRY BLUEBIRD/3.jpg"
pred = prediction(path)

if pred == 0:
    result = "ABBOTTS BABBLER"
elif pred == 1:
    result = "BAIKAL TEAL"
elif pred == 2:
    result = "CAATINGA CACHOLOTE"
elif pred == 3:
    result = "DARJEELING WOODPECKER"
elif pred == 4:
    result = "EASTERN BLUEBIRD"
elif pred == 5:
    result = "FAIRY BLUEBIRD"
elif pred == 6:
    result = "GAMBELS QUAIL"
elif pred == 7:
    result = "HAMERKOP"
elif pred == 8:
    result = "IBERIAN MAGPIE"
elif pred == 9:
    result = "JABIRU"
elif pred == 10:
    result = "KAGU"
elif pred == 11:
    result = "LARK BUNTING"
elif pred == 12:
    result = "MAGPIE GOOSE"
elif pred == 13:
    result = "NICOBAR PIGEON"
elif pred == 14:
    result = "OCELLATED TURKEY"

print(result)


    # """""##
    # 'ABBOTTS BABBLER': 0,
    # 'BAIKAL TEAL': 1,
    # 'CAATINGA CACHOLOTE': 2,
    # 'DARJEELING WOODPECKER': 3,
    # 'EASTERN BLUEBIRD': 4,
    # 'FAIRY BLUEBIRD': 5,
    # 'GAMBELS QUAIL': 6,
    # 'HAMERKOP': 7,
    # 'IBERIAN MAGPIE': 8,
    # 'JABIRU': 9,
    # 'KAGU': 10,
    # 'LARK BUNTING': 11,
    # 'MAGPIE GOOSE': 12,
    # 'NICOBAR PIGEON': 13,
    # 'OCELLATED TURKEY': 14""""