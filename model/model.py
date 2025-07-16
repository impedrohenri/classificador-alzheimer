from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
from flask import jsonify
import numpy as np
import os
from PIL import Image
import io

model = load_model('model/best_model.h5')



def model_predict(file):

    class_indices = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
    # Lê a imagem do arquivo
    img = Image.open(io.BytesIO(file.read())).convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Faz a predição
    pred = model.predict(img_array)
    class_index = np.argmax(pred, axis=1)[0]
    confidence = float(np.max(pred))
    
    return jsonify({
            'classe_predita': class_indices[class_index],
            'confianca': round(confidence, 4)
        })