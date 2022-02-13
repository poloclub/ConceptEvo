import json
import requests
from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return 'NeuEvo server to handle data'


@app.route('/send_post', methods=['GET'])
def send_post():
    params = {
        "param1": "test1",
        "param2": 123
    }
    res = requests.post(
        "http://127.0.0.1:3000/handle_post", 
        data=json.dumps(params)
    )
    print('???')
    return res.text


if __name__ == '__main__':
    print('Host the server at http://localhost:5000')
    app.run(debug=True)