# Wine Quality Prediction App 🍷

## Description

This project is a machine learning-powered web application built with Streamlit that predicts whether a wine is of **Good** or **Bad** quality based on its chemical properties. The prediction model is trained on the popular Wine Quality dataset and uses a Random Forest classifier for accuracy.

---

## Features

- **Data Exploration:** View dataset details and filter wines by quality.
- **Interactive Visualizations:** Quality distribution, alcohol content comparisons, and correlation heatmap.
- **Real-Time Prediction:** Input chemical properties and get instant quality predictions.
- **Model Performance:** Displays accuracy, confusion matrix, and classification reports.
- **User-Friendly Interface:** Clean sidebar navigation and styled UI for easy use.

---

## Dataset

The project uses the Wine Quality dataset from the UCI Machine Learning Repository:

- File: `winequality-red.csv`
- Contains chemical attributes like acidity, sugar, pH, alcohol content, and a quality rating.
- Target: Binary classification where quality >= 6 is considered "Good" and below is "Bad".

---

## Technologies Used

- Python 3.x
- Streamlit
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- pickle (for model serialization)

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/wine-quality-prediction.git
   cd wine-quality-prediction
   ```

2. **Create and activate a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure the dataset and model files are in the proper folders:**
   - `data/winequality-red.csv`
   - `model_pickle_2`

---

## Usage

Run the Streamlit app locally:

```bash
streamlit run app.py
```
