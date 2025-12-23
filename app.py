import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    with open('laptop_model.pkl', 'rb') as f:
        return pickle.load(f)

data = load_model()
model = data['model']
encoders = data['encoders']
metrics = data['metrics']

# Header
st.title("💻 Laptop Price Predictor")
st.markdown("Prediksi harga laptop berdasarkan spesifikasi menggunakan Random Forest")

# Sidebar
st.sidebar.header("📊 Model Performance")
st.sidebar.metric("MAE", f"€{metrics['mae']:.2f}")
st.sidebar.metric("RMSE", f"€{metrics['rmse']:.2f}")
st.sidebar.metric("R² Score", f"{metrics['r2']:.4f}")

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Spesifikasi Laptop")
    
    company = st.selectbox("Brand", encoders['Company'].classes_)
    typename = st.selectbox("Type", encoders['TypeName'].classes_)
    inches = st.slider("Screen Size (inches)", 10.0, 20.0, 15.6, 0.1)
    ram = st.slider("RAM (GB)", 4, 64, 8, 4)
    weight = st.slider("Weight (kg)", 0.5, 5.0, 2.0, 0.1)
    
with col2:
    st.subheader("Spesifikasi Lanjutan")
    
    cpu = st.selectbox("CPU", encoders['Cpu'].classes_)
    gpu = st.selectbox("GPU", encoders['Gpu'].classes_)
    opsys = st.selectbox("Operating System", encoders['OpSys'].classes_)
    touchscreen = st.checkbox("Touchscreen")
    ips = st.checkbox("IPS Display")

# Predict button
if st.button("🔮 Prediksi Harga", type="primary", use_container_width=True):
    # Prepare input
    input_data = pd.DataFrame({
        'Company': [encoders['Company'].transform([company])[0]],
        'TypeName': [encoders['TypeName'].transform([typename])[0]],
        'Inches': [inches],
        'Ram': [ram],
        'Weight': [weight],
        'Cpu': [encoders['Cpu'].transform([cpu])[0]],
        'Gpu': [encoders['Gpu'].transform([gpu])[0]],
        'OpSys': [encoders['OpSys'].transform([opsys])[0]],
        'Touchscreen': [1 if touchscreen else 0],
        'IPS': [1 if ips else 0]
    })
    
    # Predict
    prediction = model.predict(input_data)[0]
    
    # Display result
    st.success("### Hasil Prediksi")
    st.metric(
        label="Harga Prediksi",
        value=f"€{prediction:.2f}",
        delta=f"±€{metrics['mae']:.2f}"
    )
    
    # Visualization
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Price (Euro)"},
        gauge={
            'axis': {'range': [None, 5000]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 1000], 'color': "lightgray"},
                {'range': [1000, 2500], 'color': "gray"},
                {'range': [2500, 5000], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': prediction
            }
        }
    ))
    
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**Catatan:** Model ini dilatih menggunakan Random Forest Regressor dengan data laptop dari Kaggle")