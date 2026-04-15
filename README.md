# Insurance-Cost-Prediction-with-Feature-Engineering

# 🧠 Linear Regression on Insurance Dataset

This project demonstrates **Linear Regression** using Python and Scikit-learn to predict **insurance charges** based on customer attributes like age, BMI, smoking status, gender, and region.

---

## 📌 Project Overview

The goal of this project is to:

* Perform **Exploratory Data Analysis (EDA)**
* Preprocess categorical variables
* Train a **Linear Regression model**
* Evaluate model performance using **R² and Adjusted R²**
* Compare performance with **One-Hot Encoding**

---

## 📂 Dataset

The dataset used: **insurance.csv**

Features:

* `age` – Age of primary beneficiary
* `sex` – Gender (male/female)
* `bmi` – Body mass index
* `children` – Number of dependents
* `smoker` – Smoking status
* `region` – Residential region
* `charges` – Medical insurance cost (Target Variable)

---

## ⚙️ Tech Stack

* Python
* Pandas
* Seaborn
* Scikit-learn
* NumPy

---

## 📊 Exploratory Data Analysis

Scatter plot used to visualize relationship:

* BMI vs Charges
* Colored by Smoker status

```python
sns.scatterplot(x=insurance_data["bmi"],
                y=insurance_data["charges"],
                hue=insurance_data["smoker"])
```

---

## 🧹 Data Preprocessing

### Label Encoding

* `sex` → male = 0, female = 1
* `smoker` → no = 0, yes = 1

### One Hot Encoding

Region column converted using:

```python
pd.get_dummies(X, columns=["region"])
```

---

## 🤖 Model Training

Train-test split:

* 80% Training
* 20% Testing

Model used:

```python
LinearRegression()
```

---

## 📈 Model Evaluation

Metrics used:

* R² Score
* Adjusted R² Score

```python
r2 = r2_score(y_test, y_pred)

adjusted_r2 = 1 - ((1-r2)*(n-1)/(n-p-1))
```

---

## 🚀 Results

The model was trained twice:

1. Without One-Hot Encoding
2. With One-Hot Encoding (Improved performance)

Performance evaluated using:

* R² Score
* Adjusted R² Score

---

## 📁 Project Structure

```
Linear-Regression-Insurance/
│
├── Linear_Regression.py
├── insurance.csv
└── README.md
```

---

## ▶️ How to Run

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/linear-regression-insurance.git
```

### 2. Install Dependencies

```bash
pip install pandas seaborn scikit-learn
```

### 3. Run Script

```bash
python Linear_Regression.py
```

---

## 💡 Key Learnings

* Linear Regression Implementation
* Feature Encoding Techniques
* Train-Test Split
* Model Evaluation Metrics
* One Hot Encoding impact

---

## 📌 Future Improvements

* Add Polynomial Regression
* Try Ridge/Lasso Regression
* Add Feature Scaling
* Build Streamlit Web App
* Deploy Model

---
