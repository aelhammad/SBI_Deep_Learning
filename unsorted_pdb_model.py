import torch
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import pickle
from torch.optim.lr_scheduler import StepLR


# Asumiendo que features_array es un array de NumPy con tus características extraídas,
# pdb_names es una lista de identificadores PDB correspondientes,
# y affinities es un array de NumPy con la afinidad de cada complejo.
with open('train_features.pkl', 'rb') as f:
    train_features = pickle.load(f)
df = pd.DataFrame(train_features).transpose()
#df = pd.read_csv('/Users/allalelhommad/PYT/SBPYT_project/sample_dataframe.csv')
df.set_index('PDB_ID', inplace=True)
# # Fill the NA values
df['Psi_angle'].fillna(df['Psi_angle'].mean(), inplace=True)
df['Phi_angle'].fillna(df['Phi_angle'].mean(), inplace=True)
nan_in_df = df.isna().any()
print(nan_in_df)
# Drop the 'PDB_ID' column
X = df.drop(columns='In_Pocket')
cols_to_convert = ['Psi_angle', 'Phi_angle', 'Total_contact','Solvent_accessibility']
X[cols_to_convert] = X[cols_to_convert].apply(pd.to_numeric, errors='coerce')
# Convertir cada lista en 'Residue_Name' en un tensor de PyTorch
X['Residue_Name'] = X['Residue_Name'].apply(lambda x: x.clone().detach())
X['Secondary_structure'] = X['Secondary_structure'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
# Eliminar las filas con elementos no numéricos en 'Secondary_structure'
X = X[X['Secondary_structure'].notna()]
# Apilar todos los tensores en 'Secondary_structure' en un solo tensor
secondary_structure_tensor = torch.stack(X['Secondary_structure'].tolist())
residue_name_tensor = torch.stack(X['Residue_Name'].tolist())

# Convertir los datos numéricos a tensores
numeric_data = X[cols_to_convert].values  # cols_to_convert es tu lista de columnas numéricas
numeric_tensor = torch.tensor(numeric_data, dtype=torch.float)

# Asegurarse de que todos los tensores tienen la misma longitud en la dimensión 0
assert numeric_tensor.shape[0] == secondary_structure_tensor.shape[0] == residue_name_tensor.shape[0]

# Concatenar los tensores a lo largo de la dimensión 1
X_tensor = torch.cat((numeric_tensor, secondary_structure_tensor, residue_name_tensor), dim=1)

# Convert the data to PyTorch tensors
y = df['In_Pocket']
y = y.astype(int)  # Convert to integer dtype

y_tensor = torch.tensor(y.values, dtype=torch.float32)
print(y_tensor.shape)
# Split the data into training and validation sets

X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.33, random_state=42)

print(X_train.shape)
class MyModel(nn.Module):
    def __init__(self, input_size):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Output layer, assuming binary classification

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # No activation function for the output layer
        return x

# Initialize your model
input_size = X_train.shape[1]
model = MyModel(input_size)

# Define your loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with learning rate 0.001

# Define your dataloaders for training and validation
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=32)

# Define your optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Define the initial learning rate

# Define your learning rate scheduler
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # Reduce the learning rate by half every 10 epochs

# Training loop
num_epochs = 30
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    train_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)  # Forward pass
        labels = labels.view(-1, 1)
        loss = criterion(outputs.squeeze(), labels.squeeze())  # Calculate the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        train_loss += loss.item() * inputs.size(0)
    train_loss /= len(train_loader.dataset)
    
    # Validation loop
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0  # Initialize total to 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)  # Calculate the loss
            val_loss += loss.item() * inputs.size(0)
            predicted = torch.round(torch.sigmoid(outputs))  # Round to get binary predictions
            correct += torch.eq(predicted, labels.view_as(predicted)).sum().item()  # Compare predicted with labels
            total += labels.size(0)  # Increment total by batch size

    val_loss /= total
    val_accuracy = correct / total

    
    # Print training/validation statistics
    print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2%}')

# Evaluate the model on the test set
# Set the model to evaluation mode
model.eval()

# Disable gradient calculation
with torch.no_grad():
    # Forward pass to get the outputs
    test_outputs = model(X_test)
    
    # Calculate the loss
    test_loss = criterion(test_outputs.squeeze(), y_test)
    
    # Apply sigmoid activation to the outputs and round to get binary predictions
    predicted_test = torch.round(torch.sigmoid(test_outputs))
    
    # Calculate the number of correct predictions
    correct_predictions = (predicted_test == y_test.unsqueeze(1)).sum().item()
    
    # Calculate the total number of samples in the test set
    total_samples = len(y_test)
    
    # Calculate the test accuracy
    test_accuracy = correct_predictions / total_samples
    
    # Print test loss and accuracy
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2%}')
