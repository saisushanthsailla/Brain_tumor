# 🧠 Brain Tumor MRI Classification

This project is a machine learning based image classification system that classifies brain MRI images into four categories:

* **Glioma**
* **Meningioma**
* **No Tumor**
* **Pituitary**

I built this project as part of my Data Science training to understand how image preprocessing, feature extraction, and traditional machine learning algorithms can be applied to medical image classification.

The project also includes a **Streamlit web application** where an MRI image can be uploaded and classified using the trained model.

---

## 📌 Project Overview

Instead of directly using a deep learning model, I started with traditional machine learning and experimented with different image preprocessing and feature extraction techniques.

The main workflow is:

```text
MRI Image
    ↓
Grayscale Conversion
    ↓
CLAHE
    ↓
Image Resizing
    ↓
HOG Feature Extraction
    ↓
StandardScaler
    ↓
SVM
    ↓
Brain Tumor Class
```

The final model uses **HOG features with a tuned RBF SVM classifier**.

---

## 📂 Dataset

The project uses the **Brain Tumor MRI Dataset** from Kaggle.

Dataset:
`masoudnickparvar/brain-tumor-mri-dataset`

The dataset contains four classes:

```text
Training/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/

Testing/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/
```

The dataset already provides separate training and testing folders, so I kept the original train/test split.

---

## 🔧 Technologies Used

### Programming

* Python

### Image Processing

* OpenCV
* Scikit-image

### Machine Learning

* Scikit-learn
* SVM
* Logistic Regression
* PCA
* GridSearchCV

### Data Processing

* NumPy

### Model Saving

* Joblib

### Deployment / UI

* Streamlit

---

## 🔬 Experiments

I tested different approaches step by step to understand which preprocessing and feature extraction techniques worked best.

| Experiment                              |   Accuracy |
| --------------------------------------- | ---------: |
| Baseline + Logistic Regression          |     79.00% |
| Grayscale + Logistic Regression         |     79.06% |
| Grayscale + CLAHE + Logistic Regression |     80.13% |
| HOG + Logistic Regression               |     83.88% |
| HOG + SVM                               |     88.13% |
| Baseline + PCA                          |     77.06% |
| **HOG + Tuned SVM**                     | **90.25%** |

The final model achieved **90.25% accuracy on the test set**.

---

## 🧪 Image Preprocessing

### 1. Grayscale

The original MRI images contain three image channels. Since normal RGB color information is not the main focus for these grayscale-style MRI images, I converted the images to grayscale.

This reduced the number of raw pixel features from:

```text
100 × 100 × 3 = 30,000
```

to:

```text
100 × 100 = 10,000
```

---

### 2. CLAHE

I used **Contrast Limited Adaptive Histogram Equalization (CLAHE)** to improve local contrast.

CLAHE helps make subtle intensity differences and local structures more visible.

The parameters used were:

```python
clipLimit=2.0
tileGridSize=(8, 8)
```

---

### 3. HOG

I used **Histogram of Oriented Gradients (HOG)** to extract structural information from the MRI images.

Instead of relying directly on every pixel, HOG captures information related to:

* Edges
* Gradients
* Local shapes
* Structural patterns

The HOG configuration was:

```python
orientations=9
pixels_per_cell=(8, 8)
cells_per_block=(2, 2)
block_norm="L2-Hys"
```

This produced **4,356 features per image**.

---

## 🤖 Model

I initially experimented with Logistic Regression and then moved to SVM.

The final classifier is:

```python
SVC(kernel="rbf")
```

I used `GridSearchCV` to tune the SVM parameters.

### Best parameters

```text
C = 10
gamma = scale
kernel = rbf
```

The tuned model achieved:

```text
Test Accuracy: 90.25%
```

---

## 📊 Final Classification Results

```text
              precision    recall  f1-score   support

glioma           0.94      0.71      0.81       400
meningioma       0.85      0.92      0.88       400
notumor          0.88      1.00      0.94       400
pituitary        0.96      0.98      0.97       400

accuracy                             0.90      1600
macro avg        0.91      0.90      0.90      1600
weighted avg     0.91      0.90      0.90      1600
```

One thing I noticed during the experiments was that **glioma recall was consistently lower than the other classes**. Hyperparameter tuning improved it from **0.66 to 0.71**, but it is still the class where the model misses more samples.

---

## 🌐 Streamlit Application

I created a Streamlit interface for the trained model.

The application allows the user to:

1. Upload an MRI image
2. View the uploaded image
3. Apply the same preprocessing used during training
4. Extract HOG features
5. Run the trained SVM model
6. Display the predicted class
7. Display the model's decision scores

The four possible predictions are:

```text
Glioma
Meningioma
No Tumor
Pituitary
```

---

## 📁 Project Structure

```text
Brain_tumor/
│
├── app.py
├── brain_tumor_hog_svm.pkl
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone <your-github-repository-url>
```

Move into the project directory:

```bash
cd Brain_tumor
```

Create a Conda environment:

```bash
conda create -n brain_tumor python=3.13
```

Activate it:

```bash
conda activate brain_tumor
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

The application will open in your browser.

---

## 💾 Model

The trained model is saved using Joblib:

```python
joblib.dump(model, "brain_tumor_hog_svm.pkl")
```

The saved model contains the trained preprocessing pipeline components used by the classifier, while the image-specific preprocessing and HOG extraction are performed in the Streamlit application before prediction.

---

## 📚 What I Learned

Through this project, I got practical experience with:

* Image preprocessing using OpenCV
* Grayscale conversion
* CLAHE
* HOG feature extraction
* Dimensionality reduction using PCA
* Logistic Regression
* Support Vector Machines
* RBF kernels
* SVM hyperparameter tuning
* Cross-validation
* Model evaluation using precision, recall and F1-score
* Saving ML models using Joblib
* Building a Streamlit application
* Connecting a machine learning model to a user interface

One of the main things I learned from the experiments was that **choosing a good representation of the image can hav**
