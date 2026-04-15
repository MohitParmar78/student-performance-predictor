# 🎓 Student Exam Performance Predictor
### End-to-End Machine Learning Web Application

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/CatBoost-yellow?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
</p>

> **Predict a student's math exam score** based on socio-demographic and academic preparation factors — with a full ML pipeline from raw data to live web inference.

---

## 🔍 Problem Statement

Can we predict how well a student will perform on a math exam before they even sit for it?

This project builds a **regression model** that estimates a student's math score using features like gender, ethnicity, parental education, lunch type, and test preparation status — uncovering the real-world factors that influence academic outcomes.

---

## 🏗️ Project Architecture

```
MLprojects/
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py       # Reads & splits raw dataset
│   │   ├── data_transformation.py  # Feature engineering & preprocessing
│   │   └── model_trainer.py        # Trains & evaluates multiple models
│   │
│   ├── pipeline/
│   │   ├── predict_pipeline.py     # Inference pipeline + CustomData class
│   │   └── train_pipeline.py       # End-to-end training orchestration
│   │
│   └── utils.py                    # Shared helpers (model saving, evaluation)
│
├── notebooks/                      # EDA & experimentation notebooks
├── artifacts/                      # Saved models & preprocessor objects
├── templates/                      # Flask HTML templates
├── app.py                          # Flask web application entry point
├── setup.py
└── requirements.txt
```

---

## ✨ Features

| Feature | Detail |
|---|---|
| 🔄 **Full ML Pipeline** | Data ingestion → transformation → training → inference, fully modular |
| 🤖 **Multi-Model Training** | Evaluates multiple algorithms (CatBoost, XGBoost, Random Forest, Ridge, etc.) and auto-selects the best |
| 🧪 **Feature Engineering** | Custom `ColumnTransformer` for one-hot encoding + standard scaling |
| 🌐 **Live Web App** | Flask-based UI for real-time predictions |
| 💾 **Model Persistence** | Trained model & preprocessor saved as `.pkl` artifacts |
| 📓 **EDA Notebooks** | In-depth exploratory analysis of the dataset |

---

## 🤖 Models Evaluated

The training pipeline benchmarks the following algorithms and picks the best performer by R² score:

- **CatBoost Regressor**
- **XGBoost Regressor**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **Ridge Regression**
- **Lasso Regression**
- **Decision Tree Regressor**
- **AdaBoost Regressor**

---

## 📊 Input Features

| Feature | Type | Description |
|---|---|---|
| `gender` | Categorical | Student's gender |
| `race/ethnicity` | Categorical | Ethnic background (groups A–E) |
| `parental_level_of_education` | Categorical | Highest education level of parent |
| `lunch` | Categorical | Standard or free/reduced lunch |
| `test_preparation_course` | Categorical | Completed or none |
| `reading_score` | Numerical | Score in reading (0–100) |
| `writing_score` | Numerical | Score in writing (0–100) |

**Target:** `math_score` (continuous, 0–100)

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/MohitParmar78/MLprojects.git
cd MLprojects

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install as a package
pip install -e .
```

### Train the Model

```bash
python src/pipeline/train_pipeline.py
```

This will:
1. Ingest and split the dataset
2. Apply preprocessing transformations
3. Train & evaluate all models
4. Save the best model + preprocessor to `artifacts/`

### Run the Web App

```bash
python app.py
```

Then open your browser at: **http://localhost:5000**

---

## 🖥️ Web Application

The Flask app provides a simple form-based interface where you:
1. Enter student information (demographics + reading/writing scores)
2. Click **Predict**
3. Get an instant predicted **Math Score**

The prediction flows through the same `PredictPipeline` used in training, ensuring consistency between training-time preprocessing and inference.

---

## 📁 Artifacts

After training, the `artifacts/` directory contains:
- `model.pkl` — Best trained regression model
- `preprocessor.pkl` — Fitted `ColumnTransformer` for consistent feature encoding

---

## 🔑 Key Engineering Decisions

- **`CustomData` class** maps raw form input to a pandas DataFrame, keeping inference code clean and decoupled from Flask routes.
- **Pipeline-based preprocessing** ensures zero train-serve skew — the same preprocessor object is serialized and reloaded at inference time.
- **Modular component design** means each stage (ingestion, transformation, training) is independently testable and replaceable.

---

## 📈 What I Learned

- Building truly end-to-end ML systems beyond just model training
- Designing modular, reusable ML components in Python
- Managing the training-inference gap with serialized preprocessors
- Structuring production-style Python ML projects (`setup.py`, `src/` layout)

---

## 🙋 Author

**Mohit Parmar**
- GitHub: [@MohitParmar78](https://github.com/MohitParmar78)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
