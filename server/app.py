from flask import Flask, render_template, request, redirect, jsonify
from flask_cors import CORS
import os
import random

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def api_upload():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    print(f"Uploaded {file.filename} successfully")

    return jsonify({
        "input_image": f"/static/uploads/{file.filename}"
    })


test_images = ["/static/images/emoji1.png", "/static/images/emoji2.png",
               "/static/images/emoji3.png", "/static/images/emoji4.png", "/static/images/emoji5.png"]


@app.route('/api/emoji', methods=['GET'])
def api_emoji():
    print("Returned generated emoji successfully")
    return jsonify({
        "emoji_image": test_images[int(round(5 * random.random(), 1))]
    })


if __name__ == "__main__":
    app.run(debug=True)
