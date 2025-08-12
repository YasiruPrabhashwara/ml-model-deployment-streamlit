import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Paths
DATA_PATH = 'data/winequality-red.csv'
MODEL_PATH = 'model_pickle_2'

# Helper function to convert all columns to numeric dtype (coerce errors)
def convert_all_to_numeric(df):
    df = df.copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# Force all columns to standard NumPy dtypes (float64 or int64) for pyarrow compatibility
def force_numpy_dtypes(df):
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col].dtype):
            df[col] = df[col].astype('float64')
        elif pd.api.types.is_integer_dtype(df[col].dtype):
            df[col] = df[col].astype('int64')
        else:
            # Convert anything else to string to avoid issues
            df[col] = df[col].astype(str)
    return df

# Load data
dataset = pd.read_csv(DATA_PATH)

# Convert all columns to numeric (coerce errors)
dataset = convert_all_to_numeric(dataset)

# Prepare target column
dataset['target'] = (dataset['quality'] >= 6).astype(int)

# Fill NaNs with 0 and cast target and quality to int64
dataset['target'] = dataset['target'].fillna(0).astype('int64')
dataset['quality'] = dataset['quality'].fillna(0).astype('int64')

X = dataset.drop(['quality', 'target'], axis=1)
y = dataset['target']

train_data, test_data, train_target, test_target = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load model
with open(MODEL_PATH, 'rb') as f:
    model_2 = pickle.load(f)

# Prediction helper
def wine_prediction(input_data):
    input_df = pd.DataFrame([{
        'fixed acidity': input_data[0],
        'volatile acidity': input_data[1],
        'citric acid': input_data[2],
        'residual sugar': input_data[3],
        'chlorides': input_data[4],
        'free sulfur dioxide': input_data[5],
        'total sulfur dioxide': input_data[6],
        'density': input_data[7],
        'pH': input_data[8],
        'sulphates': input_data[9],
        'alcohol': input_data[10]
    }])
    predicted_class = model_2.predict(input_df)[0]
    predicted_prob = model_2.predict_proba(input_df)[0][predicted_class]
    quality_label = "Good" if predicted_class == 1 else "Bad"
    return quality_label, predicted_prob

# Sidebar style
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #2c3e50;
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: #f1c40f;
        }
        .css-1aumxhk, .css-16huue1 {
            color: white !important;
            font-weight: 500;
        }
        div[role='radiogroup'] label[data-baseweb='radio'] {
            background-color: #34495e;
            border-radius: 6px;
            padding: 4px 8px;
            margin-bottom: 4px;
        }
        div[role='radiogroup'] label[data-baseweb='radio']:hover {
            background-color: #3d566e;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Wine Quality Prediction")
menu = st.sidebar.radio(
    "Navigation",
    ["Project Overview", "Data Exploration", "Visualisation", "Model Prediction", "Model Performance"]
)

# Pages
if menu == "Project Overview":
    st.title("Wine Quality Prediction App")
    st.write("""
        This Streamlit app predicts whether wine quality is **Good** or **Bad**  
        based on chemical properties such as acidity, sugar, pH, alcohol content, etc.
        
        **Sections:**
        - **Data Exploration:** View and filter the dataset.
        - **Visualisation:** Interactive charts and plots.
        - **Model Prediction:** Try predicting wine quality.
        - **Model Performance:** See model accuracy, confusion matrix, and comparisons.
    """)

elif menu == "Data Exploration":
    st.header("Dataset Overview")
    st.write(f"Shape of dataset: {dataset.shape}")
    st.write(f"Columns: {list(dataset.columns)}")

    # Force numpy dtypes before displaying to avoid pyarrow error
    st.write(force_numpy_dtypes(dataset.dtypes.to_frame(name='dtype')))

    st.subheader("Sample Data")
    st.dataframe(force_numpy_dtypes(dataset.head()))

    st.subheader("Filter Data")
    quality_filter = st.slider("Select Quality", int(dataset['quality'].min()), int(dataset['quality'].max()), (3,8))
    filtered_df = dataset[(dataset['quality'] >= quality_filter[0]) & (dataset['quality'] <= quality_filter[1])]
    st.write(f"Filtered Data Shape: {filtered_df.shape}")
    st.dataframe(force_numpy_dtypes(filtered_df))

elif menu == "Visualisation":
    st.header("Data Visualisations")

    st.subheader("Wine Quality Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=dataset, x='quality', hue='quality', palette='viridis', legend=False, ax=ax)
    st.pyplot(fig)

    st.subheader("Alcohol vs Quality")
    fig, ax = plt.subplots()
    sns.boxplot(data=dataset, x='quality', y='alcohol', hue='quality', palette='coolwarm', legend=False, ax=ax)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10,6))
    corr = force_numpy_dtypes(dataset).corr()
    sns.heatmap(corr, annot=False, cmap='Blues', ax=ax)
    st.pyplot(fig)

elif menu == "Model Prediction":
    st.header("Predict Wine Quality")
    st.write("Enter the chemical properties of the wine:")

    fixed_acidity = st.number_input('Fixed Acidity', min_value=0.0)
    volatile_acidity = st.number_input('Volatile Acidity', min_value=0.0)
    citric_acid = st.number_input('Citric Acid', min_value=0.0)
    residual_sugar = st.number_input('Residual Sugar', min_value=0.0)
    chlorides = st.number_input('Chlorides', min_value=0.0)
    free_sulfur_dioxide = st.number_input('Free Sulfur Dioxide', min_value=0.0)
    total_sulfur_dioxide = st.number_input('Total Sulfur Dioxide', min_value=0.0)
    density = st.number_input('Density', min_value=0.0)
    pH = st.number_input('pH', min_value=0.0)
    sulphates = st.number_input('Sulphates', min_value=0.0)
    alcohol = st.number_input('Alcohol', min_value=0.0)

    if st.button("Predict Quality"):
        label, prob = wine_prediction([
            fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
            chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
            density, pH, sulphates, alcohol
        ])
        st.success(f"{label} (Confidence: {prob:.2f})")

elif menu == "Model Performance":
    st.header("Model Performance")

    y_pred = model_2.predict(test_data)
    acc = accuracy_score(test_target, y_pred)
    st.write(f"**Accuracy:** {acc:.2f}")

    st.text("Classification Report:")
    st.text(classification_report(test_target, y_pred))

    cm = confusion_matrix(test_target, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good'], ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    st.subheader("Model Comparison (Example)")
    st.write("Random Forest Classifier performed best among tested models.")
