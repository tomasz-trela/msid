import torch
from scripts.fetch_data import fetch_obesity_data
import pandas as pd
import numpy as np
import time
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler

print(f"CUDA dostępny: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Urządzenie CUDA: {torch.cuda.get_device_name(0)}")


try:
    df = fetch_obesity_data()
    print(f"Pomyślnie wczytano dane. Liczba wierszy: {len(df)}, liczba kolumn: {len(df.columns)}")
    print(f"Nazwy kolumn: {df.columns.tolist()}")
    print(f"Przykładowe dane (5 pierwszych wierszy):")
    print(df.head(5))
except Exception as e:
    print(f"Wystąpił błąd podczas wczytywania pliku: {e}")
    exit(1)


def fetch_obesity_data():
    return df

numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
ordinal_columns = ["CAEC", "CALC"]
binary_columns = ["Gender", "family_history_with_overweight", "FAVC", "SMOKE", "SCC"]
categorical_columns = ["MTRANS"]

ordinal_labels = [
    ["no", "Sometimes", "Frequently", "Always"],  # CAEC
    ["no", "Sometimes", "Frequently", "Always"]   # CALC
]

print("Przetwarzanie danych...")

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['NObeyesdad'])
X = df.drop(columns=['NObeyesdad'])

print(f"Unikalne klasy: {label_encoder.classes_}")
print(f"Mapowanie klas: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")

transformer = ColumnTransformer(transformers=[
    ('binary', OrdinalEncoder(), binary_columns),
    ('ordinal', OrdinalEncoder(categories=ordinal_labels), ordinal_columns),
    ('cat', OneHotEncoder(sparse_output=False), categorical_columns),
], remainder='passthrough')

X_transformed = transformer.fit_transform(X)
print(f"Kształt danych po transformacji: {X_transformed.shape}")

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_transformed)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

input_dim = X_train.shape[1]
num_classes = len(label_encoder.classes_)
print(f"Liczba cech wejściowych: {input_dim}")
print(f"Liczba klas wyjściowych: {num_classes}")

learning_rate = 0.01
epochs = 500
batch_size = 256

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return torch.nn.functional.softmax(self.linear(x), dim=1)

def train_model(device):
    X_train_device = X_train_tensor.to(device)
    y_train_device = y_train_tensor.to(device)
    
    model = LogisticRegressionModel(input_dim, num_classes).to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    start_time = time.time()
    
    dataset = torch.utils.data.TensorDataset(X_train_device, y_train_device)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_X, batch_y in data_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.long())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(data_loader)
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
    
    training_time = time.time() - start_time
    print(f'Czas treningu: {training_time:.2f} sekund')
    
    model.eval()
    with torch.no_grad():
        X_test_device = X_test_tensor.to(device)
        y_test_device = y_test_tensor.to(device)
        
        outputs = model(X_test_device)
        _, predicted = torch.max(outputs.data, 1)
        
        test_loss = criterion(outputs, y_test_device.long())
        accuracy = (predicted == y_test_device).sum().item() / y_test_device.size(0)
        
        print(f'Test Loss: {test_loss.item():.4f}')
        print(f'Accuracy: {accuracy:.4f}')
    
    return training_time, model

print("\n--- Trening na CPU ---")
cpu_time, cpu_model = train_model('cpu')

if torch.cuda.is_available():
    print("\n--- Trening na GPU (CUDA) ---")
    gpu_time, gpu_model = train_model('cuda')
    
    speedup = cpu_time / gpu_time
    print(f"\nPodsumowanie:")
    print(f"Czas treningu na CPU: {cpu_time:.2f} sekund")
    print(f"Czas treningu na GPU: {gpu_time:.2f} sekund")
    print(f"Przyspieszenie GPU vs CPU: {speedup:.2f}x")
else:
    print("\nUWAGA: CUDA nie jest dostępna - trening na GPU nie jest możliwy.")