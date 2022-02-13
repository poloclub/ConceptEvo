import os
import json
import requests
import numpy as np
from PIL import Image
from flask import Flask, request
from src.utils.utils import load_all_2d_embedding, load_neuron_feature

app = Flask(__name__)

@app.route('/')
def home():
    return 'NeuEvo server to handle data'


@app.route('/send_emb_data', methods=['GET', 'POST'])
def send_emb_data():

    embs = load_all_2d_embedding()
    return embs, 201


@app.route('/send_neuron_feature_data', methods=['GET', 'POST'])
def send_neuron_feature_data():
    data = request.form
    selected_model, instance = data['model'], data['instance']
    dir_path = f'./data/neuron_feature/{selected_model}/data/'
    img_paths = []
    for i in range(15):
        path = os.path.join(dir_path, f'{instance}-{i}.jpg')
        if os.path.exists(path):
            img_paths.append(path)
    imgs = [np.array(Image.open(path)).tolist() for path in img_paths]
    return {'imgs': imgs}, 201


if __name__ == '__main__':
    app.run(debug=True)