# Employee Attrition Prediction

A Machine Learning project that predicts whether an employee is likely to **Leave** or **Stay** based on employee and workplace-related factors.

The project covers the complete ML workflow from data preprocessing and model building to hyperparameter tuning and Streamlit deployment.

## 📌 Project Overview

Employee attrition can impact workforce stability, recruitment costs, and productivity. This project uses employee-related information to build a classification model for predicting employee attrition.

The final model is deployed as an interactive **Streamlit web application**.

## 🎯 Objective

Predict whether an employee is likely to:

* **Left** — Employee is likely to leave
* **Stayed** — Employee is likely to stay

## 📊 Dataset Features

The model uses the following features:

* Age
* Gender
* Years at Company
* Job Role
* Monthly Income
* Work-Life Balance
* Job Satisfaction
* Number of Promotions
* Overtime
* Distance from Home
* Education Level
* Marital Status
* Number of Dependents
* Job Level
* Company Size
* Remote Work
* Leadership Opportunities
* Innovation Opportunities
* Company Reputation
* Employee Recognition

### Features Removed

The following columns were excluded from the modeling process:

* `Employee ID` — Identifier
* `Company Tenure` — Excluded from the modeling process
* `Performance Rating` — Excluded from the modeling process

## 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Feature Preparation
   ↓
Train-Test Split
   ↓
Ordinal & One-Hot Encoding
   ↓
Feature Scaling
   ↓
Model Training
   ↓
Model Comparison
   ↓
Hyperparameter Tuning
   ↓
Model Evaluation
   ↓
Model Serialization
   ↓
Streamlit Deployment
```

## 🧹 Data Preprocessing

### Ordinal Encoding

Ordinal encoding was applied to categorical features with an inherent order:

* Work-Life Balance
* Job Satisfaction
* Education Level
* Job Level
* Company Size
* Company Reputation
* Employee Recognition

### One-Hot Encoding

One-hot encoding was used for nominal categorical features such as:

* Gender
* Job Role
* Overtime
* Marital Status
* Remote Work
* Leadership Opportunities
* Innovation Opportunities

`handle_unknown="ignore"` was used with `OneHotEncoder` to handle unseen categories during prediction.

### Feature Scaling

`StandardScaler` was used for models such as Logistic Regression and KNN where feature scaling is important.

## 🤖 Models Evaluated

The following classification algorithms were experimented with:

* K-Nearest Neighbors
* Logistic Regression
* Naive Bayes
* Decision Tree
* Random Forest

The Decision Tree initially showed overfitting, with significantly higher training accuracy than test accuracy.

GridSearchCV was then used to tune the model hyperparameters.

## 🌳 Random Forest

Random Forest was selected for further optimization.

### Best Parameters

```text
max_depth = 10
max_features = sqrt
min_samples_leaf = 10
min_samples_split = 30
n_estimators = 200
```

### Model Performance

The tuned Random Forest achieved approximately:

```text
Test Accuracy: 75%
```

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-score

## 📈 Model Evaluation

The project focuses on multiple evaluation metrics rather than accuracy alone:

```text
Accuracy
Precision
Recall
F1-Score
```

Recall was given particular attention because identifying employees who may leave can be important in an employee attrition use case.

## 💾 Model Serialization

The trained model was saved using Joblib:

```python
import joblib

joblib.dump(model, "employee_attrition_model.pkl")
```

The saved model contains the preprocessing and machine learning pipeline required for prediction.

## 🚀 Streamlit Deployment

The trained model was deployed using **Streamlit**.

The application allows users to enter employee information and receive an attrition prediction.

### Application Features

* Interactive employee input form
* Employee attrition prediction
* Leave/Stay classification
* Prediction probability
* Custom dark/grey UI
* Simple user-friendly interface

## 🛠️ Tech Stack

| Technology   | Purpose              |
| ------------ | -------------------- |
| Python       | Programming          |
| Pandas       | Data manipulation    |
| NumPy        | Numerical operations |
| Scikit-learn | Machine Learning     |
| Joblib       | Model serialization  |
| Streamlit    | Web application      |
| Git & GitHub | Version control      |

## 📦 Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Employee_attrition_prediction
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

## 📁 Project Structure

```text
Employee_attrition_prediction/
│
├── app.py
├── employee_attrition_model.pkl
├── requirements.txt
├── README.md
├── .gitignore
│
└── .streamlit/
    └── config.toml
```

## 🔗 Project Links

**Live Streamlit Application:**
<[YOUR_STREAMLIT_APP_URL](https://employeeattritionprediction-v3stammztbblw6qzpg5zjt.streamlit.app/)>

## 🚀 Future Improvements

* Feature engineering
* Advanced feature selection
* Probability threshold optimization
* Explainable AI using SHAP
* Model monitoring
* Improved data visualization
* Experimentation with boosting algorithms

## 👨‍💻 Author

**Sai Sushanth Sailla**



⭐ If you found this project useful, consider giving the repository a star.
