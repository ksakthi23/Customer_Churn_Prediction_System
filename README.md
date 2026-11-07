
# Customer Churn Prediction Using Machine Learning

A Machine Learning project that predicts whether a customer is likely to leave a company or continue using its services. The project uses Python, data preprocessing, exploratory data analysis, and machine learning evaluation techniques to identify potential customer churn.

##  Project Overview

Customer churn is a major challenge for businesses because losing existing customers can affect revenue and growth.

This project analyzes customer data and builds a Machine Learning model to predict customer churn. It helps businesses identify customers who may leave and take preventive actions to improve customer retention.

##  Project Objectives

- Analyze customer data to understand churn patterns.
- Perform exploratory data analysis (EDA).
- Clean and preprocess the dataset.
- Identify important factors affecting customer churn.
- Build a Machine Learning prediction model.
- Evaluate model performance using classification metrics.

## Key Features

- Data loading and exploration.
- Data cleaning and preprocessing.
- Exploratory Data Analysis with visualizations.
- Customer churn prediction.
- Model performance evaluation.
- Accuracy, Precision, Recall, F1-score and ROC-AUC metrics.

##  Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Exploratory data analysis |
| Scikit-learn | Machine Learning |
| Jupyter Notebook / VS Code | Development environment |

## 📂Project Workflow

1. **Data Collection:** Load the customer dataset.
2. **Data Exploration:** Understand the dataset and identify patterns.
3. **Data Visualization:** Create charts to explore customer behavior.
4. **Data Preprocessing:** Handle missing values and prepare features.
5. **Model Training:** Train a Machine Learning classification model.
6. **Model Evaluation:** Evaluate predictions using performance metrics.
7. **Prediction:** Identify customers who are likely to churn.

##  Model Performance

The model was evaluated using the following metrics:

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 80.55% |
| Precision | 65.72% |
| Recall | 55.88% |
| F1-Score | 60.40% |
| ROC-AUC | See evaluation output |

These results are from the model evaluation run. Performance may vary if the dataset, preprocessing or model configuration changes.

##  Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project folder

```bash
cd customer_churn_internship_project
```

### 3. Create a virtual environment

```bash
python3 -m venv .venv
```

### 4. Activate the environment

For macOS/Linux:

```bash
source .venv/bin/activate
```

For Windows:

```bash
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

If your project has a `requirements.txt` file, use:

```bash
pip install -r requirements.txt
```

## How to Run

Run the project scripts in the order required by your implementation:

```bash
python eda.py
python preprocessing.py
python evaluation.py
```

Replace script names if your actual project uses different filenames.

##  Evaluation Metrics

- **Accuracy:** Measures the overall percentage of correct predictions.
- **Precision:** Measures how many predicted churn customers actually churned.
- **Recall:** Measures how many actual churn customers were identified.
- **F1-Score:** Balances precision and recall.
- **ROC-AUC:** Measures the model's ability to distinguish between churn and non-churn customers.

##  Future Enhancements

- Build an interactive Streamlit dashboard.
- Add real-time customer churn prediction.
- Compare multiple Machine Learning algorithms.
- Improve model performance through hyperparameter tuning.
- Deploy the model as a web application.
- Add customer retention recommendations.

##  Developer

**Sakthi Krishnakumar**

Project: Customer Churn Prediction Using Machine Learning

GitHub: https://github.com/ksakthi23

## 📄 License

This project was developed for educational and internship purposes.
