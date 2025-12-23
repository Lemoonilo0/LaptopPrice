import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Data minimal agar app.py tidak error
data = {
    'model': RandomForestRegressor().fit([[0]*10], [0]),
    'encoders': {
        'Company': LabelEncoder().fit(['Acer', 'Apple', 'Asus', 'Dell', 'HP', 'Lenovo', 'MSI']),
        'TypeName': LabelEncoder().fit(['2 in 1 Convertible', 'Gaming', 'Netbook', 'Notebook', 'Ultrabook', 'Workstation']),
        'Cpu': LabelEncoder().fit(['Intel Core i5', 'Intel Core i7', 'AMD A9-Series', 'Other']),
        'Gpu': LabelEncoder().fit(['Intel HD Graphics', 'Nvidia GeForce', 'AMD Radeon']),
        'OpSys': LabelEncoder().fit(['Windows 10', 'macOS', 'No OS', 'Linux'])
    },
    'metrics': {'mae': 0, 'rmse': 0, 'r2': 0}
}

with open('laptop_model.pkl', 'wb') as f:
    pickle.dump(data, f)
print("File laptop_model.pkl berhasil dibuat!")