import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Page config
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
        border-radius: 10px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        with open('laptop_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("❌ Model file not found! Please run 'train_model.py' first.")
        st.stop()

data = load_model()
model = data['model']
encoders = data['encoders']
metrics = data['metrics']

# Header
st.title("💻 Laptop Price Predictor")
st.markdown("### Prediksi harga laptop berdasarkan spesifikasi menggunakan Machine Learning")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📊 Model Performance")
    
    st.metric("MAE", f"€{metrics['mae']:.2f}")
    st.metric("RMSE", f"€{metrics['rmse']:.2f}")
    st.metric("R² Score", f"{metrics['r2']:.4f}")
    
    if metrics['r2'] > 0.8:
        st.success("✅ Excellent Model")
    elif metrics['r2'] > 0.7:
        st.info("✓ Good Model")
    else:
        st.warning("⚠ Fair Model")
    
    st.markdown("---")
    st.markdown("### 📖 About")
    st.info("""
    **Algorithm:** Random Forest
    
    **Dataset:** Kaggle Laptop Price
    
    **Features:** 10+ specs
    """)

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔧 Basic Specifications")
    
    company = st.selectbox("🏢 Brand", encoders['Company'].classes_)
    typename = st.selectbox("📱 Type", encoders['TypeName'].classes_)
    inches = st.slider("📺 Screen Size (inches)", 10.0, 20.0, 15.6, 0.1)
    ram = st.select_slider("💾 RAM (GB)", options=[4, 8, 16, 32, 64], value=8)
    weight = st.slider("⚖️ Weight (kg)", 0.5, 5.0, 2.0, 0.1)

with col2:
    st.subheader("⚙️ Advanced Specifications")
    
    cpu = st.selectbox("🖥️ CPU", encoders['Cpu'].classes_)
    gpu = st.selectbox("🎮 GPU", encoders['Gpu'].classes_)
    opsys = st.selectbox("💿 Operating System", encoders['OpSys'].classes_)
    
    col_check1, col_check2 = st.columns(2)
    with col_check1:
        touchscreen = st.checkbox("👆 Touchscreen")
    with col_check2:
        ips = st.checkbox("🖼️ IPS Display")

st.markdown("---")

# Predict button
if st.button("🔮 PREDIKSI HARGA", type="primary"):
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
    st.success("### ✅ Hasil Prediksi")
    
    col_result1, col_result2, col_result3 = st.columns(3)
    
    with col_result1:
        st.metric("💰 Harga (EUR)", f"€{prediction:.2f}")
    
    with col_result2:
        st.metric("📊 Margin Error", f"±€{metrics['mae']:.2f}")
    
    with col_result3:
        rupiah = prediction * 17000
        st.metric("💵 Estimasi (IDR)", f"Rp{rupiah:,.0f}")
    
    # Visualization
    st.markdown("### 📈 Price Category")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    categories = ['Budget\n€0-1000', 'Mid-range\n€1000-2500', 'High-end\n€2500+']
    ranges = [1000, 2500, 5000]
    colors = ['#90EE90', '#FFD700', '#FF6B6B']
    
    ax.barh(categories, ranges, color=colors, alpha=0.3)
    ax.axvline(x=prediction, color='darkblue', linewidth=3, linestyle='--', 
               label=f'Prediction: €{prediction:.0f}')
    
    ax.set_xlabel('Price (Euro)', fontsize=12, fontweight='bold')
    ax.set_title('Laptop Price Category', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    st.pyplot(fig)
    
    # Category description
    if prediction < 1000:
        st.info("💚 **Budget Laptop** - Entry-level, suitable for light tasks")
    elif prediction < 2500:
        st.info("💛 **Mid-range Laptop** - Good for productivity and multimedia")
    else:
        st.info("❤️ **High-end Laptop** - Premium, for gaming/editing/workstation")
    
    # Specification summary
    with st.expander("📋 Specification Summary"):
        st.write(f"""
        - **Brand:** {company}
        - **Type:** {typename}
        - **Screen:** {inches} inches
        - **RAM:** {ram} GB
        - **Weight:** {weight} kg
        - **CPU:** {cpu}
        - **GPU:** {gpu}
        - **OS:** {opsys}
        - **Touchscreen:** {'Yes' if touchscreen else 'No'}
        - **IPS:** {'Yes' if ips else 'No'}
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p><strong>Machine Learning Project</strong> | Random Forest Regressor</p>
    <p>Dataset: Kaggle | Framework: Scikit-learn & Streamlit</p>
</div>
""", unsafe_allow_html=True)
