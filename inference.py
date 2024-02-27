from flask import Flask, request, jsonify


app = Flask(__name__)


@app.route('/prediction/', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        data = request.json['input_data']
        result = 0
        return jsonify({'prediction': result})
    else:
        return '<p>Wokring</p>'