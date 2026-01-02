import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# --- STAGE 1 & 2: DATA LOADING & MODEL OPTIMIZATION ---

@st.cache_resource
def get_trained_model():
    # Modified line to match your uploaded filename
    file_name = "../data/lungcancer.csv"
    
    # Check if file exists in the current directory
    if not os.path.exists(file_name):
        st.error(f"File '{file_name}' not found! Make sure it is in the same folder as this script.")
        return None, None, None

    # Load dataset
    data = pd.read_csv(file_name)
    
    # Clean column names (remove leading/trailing spaces)
    data.columns = data.columns.str.strip()
    
    # Encode categorical columns
    le = LabelEncoder()
    data['GENDER'] = le.fit_transform(data['GENDER'])
    data['LUNG_CANCER'] = le.fit_transform(data['LUNG_CANCER'])
    
    # Split features & target
    X = data.drop('LUNG_CANCER', axis=1)
    y = data['LUNG_CANCER']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Stage 2 Optimization: Finding the best K
    k_values = range(1, 21)
    accuracy_list = []
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
        knn.fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_test_scaled)
        accuracy_list.append(accuracy_score(y_test, y_pred))
    
    best_k = k_values[np.argmax(accuracy_list)]
    
    # Final optimized model
    knn_final = KNeighborsClassifier(n_neighbors=best_k, weights='distance')
    knn_final.fit(X_train_scaled, y_train)
    
    return knn_final, scaler, max(accuracy_list)

# Initialize model
model, scaler, accuracy = get_trained_model()

# --- STAGE 3: FORM STRUCTURE (HTML/CSS via Streamlit) ---

st.set_page_config(page_title="Lung Cancer Detector", layout="centered")
st.title("🫁 Lung Cancer Detection System")

if model:
    st.write(f"**Model Accuracy:** {accuracy*100:.2f}%")
    st.write("Please fill in the patient details to check the risk level.")

    with st.form("lung_cancer_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Gender", ["M", "F"])
            age = st.number_input("Age", 1, 100, 50)
            smoking = st.selectbox("Smoking", [1, 2], format_func=lambda x: "Yes" if x==2 else "No")
            yellow_fingers = st.selectbox("Yellow Fingers", [1, 2], format_func=lambda x: "Yes" if x==2 else "No")
            anxiety = st.selectbox("Anxiety", [1, 2], format_func=lambda x: "Yes" if x==2 else "No")
            peer_pressure = st.selectbox("Peer Pressure", [1, 2], format_func=lambda x: "Yes" if x==2 else "No")
            chronic_disease = st.selectbox("Chronic Disease", [1, 2], format_func=lambda x: "Yes" if x==2 else "No")
            fatigue = st.selectbox("Fatigue", [1, 2], format_func=lambda x: "Yes" if x==2 else "No")

        with col2:
            allergy = st.selectbox("Allergy", [1, 2], format_func=lambda x: "Yes" if x==2 else "No")
            wheezing = st.selectbox("Wheezing", [1, 2], format_func=lambda x: "Yes" if x==2 else "No")
            alcohol = st.selectbox("Alcohol Consuming", [1, 2], format_func=lambda x: "Yes" if x==2 else "No")
            coughing = st.selectbox("Coughing", [1, 2], format_func=lambda x: "Yes" if x==2 else "No")
            shortness_breath = st.selectbox("Shortness of Breath", [1, 2], format_func=lambda x: "Yes" if x==2 else "No")
            swallowing_diff = st.selectbox("Swallowing Difficulty", [1, 2], format_func=lambda x: "Yes" if x==2 else "No")
            chest_pain = st.selectbox("Chest Pain", [1, 2], format_func=lambda x: "Yes" if x==2 else "No")
        
        submit = st.form_submit_button("Analyze Results")

    if submit:
        # Convert inputs for prediction
        gen_val = 1 if gender == "M" else 0
        input_data = np.array([[gen_val, age, smoking, yellow_fingers, anxiety, peer_pressure, 
                                chronic_disease, fatigue, allergy, wheezing, alcohol, 
                                coughing, shortness_breath, swallowing_diff, chest_pain]])
        
        # Scale input and predict
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        
        # Show Result
        if prediction[0] == 1:
            st.error("### Prediction: POSITIVE - Patient is at risk.")
        else:
            st.success("### Prediction: NEGATIVE - No significant risk detected.")