# Handwritten Digit Recognition (MNIST) – Flask Deployment

A complete, end-to-end project:
- **01_EDA.ipynb** – quick data exploration
- **02_Modeling.ipynb** – preprocessing, CNN training with augmentation, evaluation, model saving
- **Flask web app** (`app.py`) – upload an image → preprocess (MNIST-style) → predict → display result

Flow: **User → Upload → Server (Preprocess & Predict) → Result → Display**

---

## Dataset
- CSV format like MNIST: `train.csv` (labels + 28×28 pixels), `test.csv` (pixels only).
- Put files in `./data/`.


mnist-digit-recognition-flask/
│
├── app.py # Flask app entry point
├── utils/ # Preprocessing & model loading
├── templates/ # HTML templates
├── static/ # CSS, JS, uploaded images
├── outputs/ # Model, plots, confusion matrix
├── notebooks/ # Jupyter notebooks (EDA, training, evaluation)
├── requirements.txt # Dependencies
├── README.md # Project documentation
└── .gitignore # Ignored files/folders



---

## How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/mnist-digit-recognition-flask.git
cd mnist-digit-recognition-flask


2.Create and activate virtual environment
-> Windows

python -m venv venv
venv\Scripts\activate

3.Install dependencies
pip install -r requirements.txt

4. Run the Flask app
python app.py

5. Open in browser
http://127.0.0.1:5000


Training Details
Dataset: MNIST

Augmentation: rotation, zoom, shift, shear

Optimizer: Adam

Loss: Categorical Crossentropy

Accuracy: ~98–99% on validation set

