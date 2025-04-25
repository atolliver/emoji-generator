from flask import Flask, render_template, request, redirect, url_for, send_from_directory, send_file, jsonify
import os
import pandas as pd
from io import BytesIO
from PIL import Image
import numpy as np
import io
import base64
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Embedding, Reshape, concatenate
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam

# Constants
basedir = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(basedir, 'static', 'uploads')
EMOJI_FOLDER = os.path.join(basedir, 'static', 'images')
DATA_PATH = os.path.join(basedir, 'full_emoji.csv') # Fix: Correct dataset path
IMAGE_SHAPE = (64, 64, 3) # Set your desired image shape

# Create upload directory if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load data (Corrected path)
try:
    data = pd.read_csv(DATA_PATH)
    print("Data loaded successfully")
except FileNotFoundError:
    print(f"Error: Could not find dataset at {DATA_PATH}")
    data = None  # Handle the case where the dataset isn't loaded
except Exception as e:
    print(f"Error loading data: {e}")
    data = None

# --- Utility Functions ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# --- Model Training ---
def create_emoji_cnn(image_shape, num_emojis, embedding_dim=128):
    # Image input branch
    image_input = Input(shape=image_shape, name='image_input')
    conv1 = Conv2D(32, (3, 3), activation='relu')(image_input)
    pool1 = MaxPooling2D((2, 2))(conv1)
    conv2 = Conv2D(64, (3, 3), activation='relu')(pool1)
    pool2 = MaxPooling2D((2, 2))(conv2)
    flat = Flatten()(pool2)

    # Emoji input branch
    emoji_input = Input(shape=(1,), name='emoji_input')
    embedding = Embedding(input_dim=num_emojis, output_dim=embedding_dim)(emoji_input)
    emoji_flat = Flatten()(embedding)

    # Concatenate image and emoji features
    merged = concatenate([flat, emoji_flat])

    # Dense layers for output
    dense1 = Dense(256, activation='relu')(merged)
    output_image = Dense(np.prod(image_shape), activation='sigmoid')(dense1)  # Output layer
    output_image = Reshape(image_shape)(output_image)  # Reshape to image dimensions

    model = Model(inputs=[image_input, emoji_input], outputs=output_image)
    return model


def train_model(model, image_data, emoji_data, epochs=10, batch_size=32):
    # Compile the model
    optimizer = Adam(learning_rate=0.001)  # You can adjust the learning rate
    model.compile(optimizer=optimizer, loss='mse')  # Mean Squared Error loss
    # Train the model
    history = model.fit([image_data, emoji_data], image_data, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    # Plot training history
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


@app.route('/generate_emoji_image', methods=['POST'])
def generate_emoji_image():
    data = request.get_json()
    input_image = np.array(data['image']).reshape((1,)+IMAGE_SHAPE) / 255.0  # Normalize
    input_emoji = np.array([data['emoji']])
    generated_image = model.predict([input_image, input_emoji])
    generated_image = (generated_image[0] * 255).astype(np.uint8)  # Scale back to 0-255
    img = Image.fromarray(generated_image)
    img_io = BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')

# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'image' not in request.files:
            return redirect(request.url)

        file = request.files['image']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('result', filename=filename))

    return render_template('index.html')

@app.route('/result')
def result():
    filename = request.args.get('filename')
    user_image_url = url_for('static', filename=f'uploads/{filename}')
    emoji_image_url = url_for('static', filename='images/emoji.png')  # Hardcoded URL

    return render_template(
        'result.html',
        user_image=user_image_url,
        emoji_image=emoji_image_url
    )

# --- Sample Training ---
if __name__ == '__main__':
    # Load and preprocess your data here (replace with your actual data loading)
    if data is not None:
        num_samples = 100  # Reduced sample size
        image_data = np.random.rand(num_samples, IMAGE_SHAPE[0], IMAGE_SHAPE[1], IMAGE_SHAPE[2])
        emoji_data = np.random.randint(0, 10, num_samples)  # Assuming 10 emojis

        # Convert emoji data to one-hot encoding
        num_emojis = len(np.unique(emoji_data))
        emoji_data = to_categorical(emoji_data, num_classes=num_emojis)

        # Split data into training and testing
        image_train, image_test, emoji_train, emoji_test = train_test_split(
            image_data, emoji_data, test_size=0.2, random_state=42
        )

        # Create and train the model
        model = create_emoji_cnn(IMAGE_SHAPE, num_emojis)
        train_model(model, image_train, emoji_train, epochs=2, batch_size=32)
    app.run(debug=True)
