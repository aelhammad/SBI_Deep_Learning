import torch
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import pickle
from torch.optim.lr_scheduler import StepLR

with open('train_features_11.pkl', 'rb') as f:
    train_features = pickle.load(f)

with open('test_features_4.pkl', 'rb') as f:
    test_features = pickle.load(f)

with open('val_features_5.pkl', 'rb') as f:
    val_features = pickle.load(f)

#with open('sample_dataframe.csv') as f:
#    train_features = pd.read_csv(f)
def preprocess_features(features):
    df = pd.DataFrame(features)
    df['Psi_angle'].fillna(df['Psi_angle'].mean(), inplace=True)
    df['Phi_angle'].fillna(df['Phi_angle'].mean(), inplace=True)
    nan_in_df = df.isna().any()

    pdb_data = {}
    for pdb_id, data in df.groupby('PDB_ID'):
        pdb_data[pdb_id] = data
        data.set_index('PDB_ID', inplace=True)

    loaders = {}
    for pdb_id, data in pdb_data.items():
        X = data.drop(columns='In_Pocket')
        cols_to_convert = ['Psi_angle', 'Phi_angle', 'Total_contact','Solvent_accessibility']
        X[cols_to_convert] = X[cols_to_convert].apply(pd.to_numeric, errors='coerce')

        X['Residue_Name'] = X['Residue_Name'].apply(lambda x: x.clone().detach())
        X['Secondary_structure'] = X['Secondary_structure'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)

        X = X[X['Secondary_structure'].notna()]

        secondary_structure_tensor = torch.stack(X['Secondary_structure'].tolist())
        residue_name_tensor = torch.stack(X['Residue_Name'].tolist())

        numeric_data = X[cols_to_convert].values
        numeric_tensor = torch.tensor(numeric_data, dtype=torch.float)

        assert numeric_tensor.shape[0] == secondary_structure_tensor.shape[0] == residue_name_tensor.shape[0]

        X_tensor = torch.cat((numeric_tensor, secondary_structure_tensor, residue_name_tensor), dim=1)

        y = data['In_Pocket']
        y = y.astype(int)

        y_tensor = torch.tensor(y.values, dtype=torch.float32)

        min_val_X = torch.min(X_tensor)
        max_val_X = torch.max(X_tensor)

        X_tensor_normalized = (X_tensor - min_val_X) / (max_val_X - min_val_X)

        min_val_y = torch.min(y_tensor)
        max_val_y = torch.max(y_tensor)

        y_tensor_normalized = (y_tensor - min_val_y) / (max_val_y - min_val_y)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        loaders[pdb_id] = loader

    return loaders

train_loaders = preprocess_features(train_features)
test_loaders = preprocess_features(test_features)
val_loaders = preprocess_features(val_features)
    
class MyModel(nn.Module):
    def __init__(self, input_size):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Capa de salida, asumiendo clasificación binaria

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # No hay función de activación para la capa de salida
        return x



if __name__ == "__main__":
    # Inicializar tu modelo
    #input_size = X_tensor.shape[1]
    input_size = 34
    model = MyModel(input_size)

    # Definir la función de pérdida y el optimizador
    criterion = nn.BCEWithLogitsLoss()  # Pérdida de entropía cruzada binaria para la clasificación binaria
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizador Adam con tasa de aprendizaje 0.001

    # Definir el programador de la tasa de aprendizaje
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # Reducir la tasa de aprendizaje a la mitad cada 10 épocas

    # Bucle de entrenamiento
    num_epochs = 30
    for epoch in range(num_epochs):
        model.train()  # Poner el modelo en modo de entrenamiento
        train_loss = 0.0
        for pdb_id, loader in train_loaders.items():
            for inputs, labels in loader:
                optimizer.zero_grad()  # Poner a cero los gradientes
                outputs = model(inputs)  # Paso hacia adelante
                labels = labels.view(-1, 1)
                loss = criterion(outputs.squeeze(), labels.squeeze())  # Calcular la pérdida
                loss.backward()  # Paso hacia atrás
                optimizer.step()  # Actualizar los pesos
                train_loss += loss.item() * inputs.size(0)
            train_loss /= len(loader.dataset)

        # Imprimir las estadísticas de entrenamiento
        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}')

        # Validación
        model.eval()  # Poner el modelo en modo de evaluación
        val_loss = 0.0
        for pdb_id, loader in val_loaders.items():
            for inputs, labels in loader:
                outputs = model(inputs)  # Paso hacia adelante
                labels = labels.view(-1, 1)
                loss = criterion(outputs.squeeze(), labels.squeeze())  # Calcular la pérdida
                val_loss += loss.item() * inputs.size(0)
            val_loss /= len(loader.dataset)

        # Imprimir las estadísticas de validación
        print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}')

        # Prueba
    # Prueba
    # Prueba
    # Prueba
    test_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    for pdb_id, loader in test_loaders.items():
        for inputs, labels in loader:
            outputs = model(inputs)  # Paso hacia adelante
            labels = labels.view(-1, 1)
            loss = criterion(outputs.squeeze(), labels.squeeze())  # Calcular la pérdida
            test_loss += loss.item() * inputs.size(0)

            # Calcular la precisión
            predicted = torch.round(outputs)  # Redondear las salidas del modelo para obtener las predicciones
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        test_loss /= len(loader.dataset)

    # Imprimir las estadísticas de prueba
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {correct_predictions / total_predictions * 100:.2f}%')

    # Guardar el modelo
    torch.save(model.state_dict(), 'model.pth')