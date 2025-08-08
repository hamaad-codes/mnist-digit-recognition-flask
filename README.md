# Handwritten Digit Recognition (MNIST) – .h5 + Flask

A complete, end-to-end project:
- **01_EDA.ipynb** – quick data exploration
- **02_Modeling.ipynb** – preprocessing, CNN training with augmentation, evaluation, model saving
- **Flask web app** (`app.py`) – upload an image → preprocess (MNIST-style) → predict → display result

Flow: **User → Upload → Server (Preprocess & Predict) → Result → Display**

---

## Dataset
- CSV format like MNIST: `train.csv` (labels + 28×28 pixels), `test.csv` (pixels only).
- Put files in `./data/`.

