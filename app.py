import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from model import model
import os

app = Flask(__name__)
CORS(app) 


@app.route('/')
def home():
    return render_template('./home.html', valor=None)

@app.route("/", methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'erro': 'Nenhuma imagem enviada'}), 400

    file = request.files['file']

    file.save('./static/temp.png')

    if file.filename == '':
        return jsonify({'erro': 'Nome de arquivo vazio'}), 400

    with open('./static/temp.png', "rb") as f:
        valor = model.model_predict(f)

    valor['img_path'] = './static/temp.png'

    return render_template("./home.html", valor=valor)


if __name__ == '__main__':
    app.run(debug=True)