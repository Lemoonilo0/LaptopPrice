import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*50)
print("LAPTOP PRICE PREDICTION - MODEL TRAINING")
print("="*50)

# Load dataset
print("\n[1/6] Loading dataset...")
try:
    df = pd.read_csv('laptop_price.csv', encoding='latin1')
    print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
except FileNotFoundError:
    print("❌ Error: File 'laptop_price.csv' not found!")
    print("Download dataset from: https://www.kaggle.com/datasets/muhammetvarl/laptop-price")
    exit()

# Data preprocessing
print("\n[2/6] Preprocessing data...")
# Remove unnecessary columns
if 'laptop_ID' in df.columns:
    df = df.drop(['laptop_ID'], axis=1)

# Handle missing values
df = df.dropna()
print(f"✓ Data cleaned: {df.shape[0]} rows remaining")

# Encode categorical features
print("\n[3/6] Encoding categorical features...")
le_company = LabelEncoder()
le_typename = LabelEncoder()
le_cpu = LabelEncoder()
le_gpu = LabelEncoder()
le_os = LabelEncoder()

df['Company'] = le_company.fit_transform(df['Company'])
df['TypeName'] = le_typename.fit_transform(df['TypeName'])
df['Cpu'] = le_cpu.fit_transform(df['Cpu'])
df['Gpu'] = le_gpu.fit_transform(df['Gpu'])
df['OpSys'] = le_os.fit_transform(df['OpSys'])

print(f"✓ Features encoded successfully")

# Features and target
X = df.drop('Price_euros', axis=1)
y = df['Price_euros']

print(f"\n[4/6] Splitting data...")
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Testing set: {X_test.shape[0]} samples")

# Train model
print(f"\n[5/6] Training Random Forest model...")
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("✓ Model training completed!")

# Evaluate
print(f"\n[6/6] Evaluating model...")
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n" + "="*50)
print("MODEL PERFORMANCE METRICS")
print("="*50)
print(f"Mean Absolute Error (MAE)  : €{mae:.2f}")
print(f"Root Mean Squared Error    : €{rmse:.2f}")
print(f"R² Score                   : {r2:.4f} ({r2*100:.2f}%)")
print("="*50)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 5 Most Important Features:")
for idx, row in feature_importance.head().iterrows():
    print(f"  {row['feature']:.<20} {row['importance']:.4f}")

# Save model and encoders
print("\n[SAVE] Saving model and encoders...")
model_data = {
    'model': model,
    'encoders': {
        'Company': le_company,
        'TypeName': le_typename,
        'Cpu': le_cpu,
        'Gpu': le_gpu,
        'OpSys': le_os
    },
    'feature_names': X.columns.tolist(),
    'metrics': {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    },
    'feature_importance': feature_importance.to_dict()
}

with open('laptop_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("✓ Model saved as 'laptop_model.pkl'")
print("\n" + "="*50)
print("✅ TRAINING COMPLETED SUCCESSFULLY!")
print("="*50)
print("\nNext steps:")
print("1. Run Streamlit app: streamlit run app.py")
print("2. Test predictions with different specifications")

print("="*50)

