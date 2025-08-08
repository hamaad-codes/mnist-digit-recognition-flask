import os
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from tensorflow.keras.models import load_model
from dotenv import load_dotenv

# ---------------------- Config ----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
MODEL_PATH = os.path.join(BASE_DIR, 'outputs', 'mnist_model.h5')  # or .keras
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg'}

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------- App & Model ----------------------
load_dotenv()

app = Flask(__name__)
app.secret_key = 'SECRET_KEY'  # for flash messages

# Load once at startup
model = load_model(MODEL_PATH)

# ---------------------- Utils ----------------------
def allowed_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def preprocess_image(path: str) -> np.ndarray:
    """
    Load image -> grayscale -> auto-invert if needed -> crop to digit ->
    pad to square -> resize to 20x20 -> pad to 28x28 -> normalize -> reshape.
    Returns (1, 28, 28, 1) float32 in [0,1].
    """
    # 1) Load grayscale
    img = Image.open(path).convert('L')
    arr = np.asarray(img).astype('float32')

    # 2) Auto-invert if background is bright (paper)
    # MNIST expects white digit on black bg.
    if arr.mean() > 127:           # white background likely
        arr = 255.0 - arr

    # 3) Normalize to [0,1]
    arr /= 255.0

    # 4) Threshold lightly to clean noise (keep as float)
        # Adaptive threshold (clean noise)
    thr = max(0.05, arr.mean()*0.6)   # dynamic cut
    arr = np.where(arr < thr, 0.0, arr)

        #  Stroke thicken (helps skinny/anti-aliased digits)
    img_bin = Image.fromarray((arr*255).astype('uint8'))
    img_bin = img_bin.filter(ImageFilter.MaxFilter(3))   # dilate
    arr = np.asarray(img_bin).astype('float32') / 255.0

    # 5) Find bounding box of the digit
    ys, xs = np.where(arr > 0.05)
    if len(xs) == 0 or len(ys) == 0:
        # fallback: just center-crop/resize
        arr = np.pad(arr, 4, mode='constant', constant_values=0.0)
        arr = Image.fromarray((arr * 255).astype('uint8')).resize((28, 28), Image.BILINEAR)
        arr = np.asarray(arr).astype('float32') / 255.0
        return arr.reshape(1, 28, 28, 1)

    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1
    crop = arr[y1:y2, x1:x2]

    # 6) Pad to square
    h, w = crop.shape
    side = max(h, w)
    pad_top = (side - h) // 2
    pad_bottom = side - h - pad_top
    pad_left = (side - w) // 2
    pad_right = side - w - pad_left
    square = np.pad(crop, ((pad_top, pad_bottom), (pad_left, pad_right)),
                    mode='constant', constant_values=0.0)

    # 7) Resize like MNIST: 20x20 content + 4px padding -> 28x28
    img20 = Image.fromarray((square * 255).astype('uint8')).resize((20, 20), Image.BILINEAR)
    arr20 = np.asarray(img20).astype('float32') / 255.0
    arr28 = np.pad(arr20, 4, mode='constant', constant_values=0.0)

    # 8) Final reshape
    return arr28.reshape(1, 28, 28, 1).astype('float32')



def predict_digit(img_array: np.ndarray) -> tuple[int, float]:
    """Return (digit, confidence in [0,1])."""
    probs = model.predict(img_array, verbose=0)[0]
    digit = int(np.argmax(probs))
    confidence = float(np.max(probs))
    return digit, confidence

# ---------------------- Routes ----------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part in the request.')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('Please choose an image file to upload.')
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash('Unsupported file type. Please upload PNG/JPG/JPEG.')
            return redirect(request.url)

        # Safe filename + timestamp to avoid collisions
        filename = secure_filename(file.filename)
        name, ext = os.path.splitext(filename)
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        saved_name = f"{name}-{timestamp}{ext}"
        save_path = os.path.join(UPLOAD_FOLDER, saved_name)
        file.save(save_path)

        # Preprocess & predict
        img_arr = preprocess_image(save_path)
        digit, conf = predict_digit(img_arr)

        # Render with result
        return render_template(
            'index.html',
            uploaded_image=url_for('static', filename=f'uploads/{saved_name}'),
            prediction=digit,
            confidence=f"{conf*100:.2f}%"
        )

    # GET
    return render_template('index.html')


if __name__ == '__main__':
    # In production, use a real WSGI server (gunicorn/uwsgi). Debug off.
    app.run(host='0.0.0.0', port=5000, debug=True)