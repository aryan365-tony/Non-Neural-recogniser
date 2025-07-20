# 🧠✍️ Non-Neural Handwriting Recognizer with GPU Acceleration

A high-performance **handwriting digit recognizer** built entirely without neural networks, leveraging **GPU-accelerated machine learning (RAPIDS cuML)** for blazing fast training and inference.

- 🚀 **Fast**: Uses NVIDIA RAPIDS (cuDF, cuML) for GPU-based SVM, Random Forest, and KNN classifiers.
- 🧠 **Non-Neural**: Relies on classical computer vision & ML—no deep learning used!
- 🔍 **Accurate**: Uses HOG + PCA + ensemble learning for robust prediction.
- 🧰 **Modular Code**: Clean Python module structure, easy to extend or deploy.



## ⚙️ Project Structure

```
handwriting_gpu_ocr/
│
├── data/                         # EMNIST will be downloaded here
├── models/                       # (Optional) Save trained models
├── src/
│   ├── config.py                 # Constants & paths
│   ├── gpu_check.py              # Verifies CUDA & RAPIDS versions
│   ├── data_loader.py            # Loads and filters EMNIST digits
│   ├── preprocessing.py          # Deskewing + Binarization
│   ├── feature_extraction.py     # HOG descriptor + PCA compression
│   ├── train.py                  # GPU-based model training
│   ├── ensemble.py               # Soft-voting logic
│   └── inference.py              # Predicts from a single image
│
├── main.py                       # Train & evaluate the ensemble model
├── demo.py                       # Visualize predictions with matplotlib
├── requirements.txt              # Dependencies (RAPIDS + others)
└── README.md                     # You're here!
```


## 🛠️ Approach

### 1. **Dataset**
- Uses EMNIST-Balanced (handwritten characters).
- Filters only **digits [0–9]** for focused digit classification.

### 2. **Preprocessing**
- **Deskewing** using image moments and affine transform.
- **Binarization** using Otsu thresholding.

### 3. **Feature Extraction**
- **Histogram of Oriented Gradients (HOG)** for shape-based features.
- **PCA** to reduce feature vector dimensionality (to 100 dims).

### 4. **GPU Training**
- Converts all features to `cudf.DataFrame` format.
- Trains 3 different **cuML models**:
  - SVC (RBF Kernel)
  - RandomForestClassifier
  - KNeighborsClassifier

### 5. **Ensemble Voting**
- Combines model predictions using **soft voting** (averaging probabilities).
- Final prediction is the class with the highest average probability.

### 6. **Inference**
- Supports fast inference on single images via recognize_image() function.



## 🖥️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/handwriting_gpu_ocr.git
cd handwriting_gpu_ocr
```

### 2. Install RAPIDS + Requirements

**✅ This project requires NVIDIA GPU + CUDA (11.5 or later)**

Use conda or pip to install:

```bash
pip install -r requirements.txt
```

Or follow [RAPIDS installation guide](https://rapids.ai/start.html) for your CUDA version.

---

## ▶️ Run the Project

### 1. Check GPU + RAPIDS versions
```bash
python -m src.gpu_check
```

### 2. Train & Evaluate Models
```bash
python main.py
```

### 3. Run Inference Demo
```bash
python demo.py
```

> You’ll see a random digit from the test set and the model’s prediction.

---

## 📦 Inference Function (Usage)

```python
from src.inference import recognize_image
predicted_digit = recognize_image(my_image, pca, scaler, svc, rf, knn)
```

- Input: Grayscale 28x28 NumPy image
- Output: Predicted digit (0–9)

---

## 🔧 Requirements

- Python 3.8+
- NVIDIA GPU + CUDA 11+
- RAPIDS libraries: `cudf`, `cuml`, `cupy`, `rmm`
- Others: `opencv`, `torch`, `scikit-image`, `scikit-learn`, `matplotlib`

See `requirements.txt` for the full list.

---

## 📚 References

- EMNIST Dataset: https://www.nist.gov/itl/products-and-services/emnist-dataset
- RAPIDS AI: https://rapids.ai
- HOG Descriptor: https://scikit-image.org/docs/dev/api/skimage.feature.html#hog

---

## 🙌 Acknowledgements

This project is inspired by traditional CV + ML pipelines, with modern GPU acceleration. A great demonstration that **you don’t always need deep learning for high accuracy**.

---

## 🧪 TODO / Enhancements

- [ ] Save + Load trained models
- [ ] Add CLI for inference from image file
- [ ] Add support for full EMNIST (letters + digits)
- [ ] Add benchmarking vs CPU models

---

## 📝 License

This project is MIT licensed. Feel free to use or extend it in your own work.

