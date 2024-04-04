import torch
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import pickle
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# Function to preprocess the features
def preprocess_features(features):
    '''
    Preprocess the features for training.

    Args:
        features (dict): Dictionary containing features.

    Returns:
        dict: Dictionary containing preprocessed features.
    '''
    df = pd.DataFrame(features)
    df['Psi_angle'].fillna(df['Psi_angle'].mean(), inplace=True)
    df['Phi_angle'].fillna(df['Phi_angle'].mean(), inplace=True)
    
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

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        loaders[pdb_id] = loader
    return loaders

# Define the neural network model
class MyModel(nn.Module):
    '''Neural network model for binary classification.'''
    def __init__(self, input_size):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Output layer, assuming binary classification

    def forward(self, x):
        '''Forward pass of the neural network.'''
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # No activation function for the output layer
        return x

if __name__ == "__main__":
    # Initialize the model
    input_size = 34
    model = MyModel(input_size)

    # Define the loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with learning rate 0.001

    # Define the learning rate scheduler
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)  # Reduce the learning rate by half every 10 epochs
    # Load train, test, and validation features from pickle files
    with open('datasets/train_features_final.pkl', 'rb') as f:
        train_features = pickle.load(f)

    with open('datasets/test_features_final.pkl', 'rb') as f:
        test_features = pickle.load(f)

    with open('datasets/val_features_final.pkl', 'rb') as f:
        val_features = pickle.load(f)
    # Preprocess the train, test, and validation features
    train_loaders = preprocess_features(train_features)
    test_loaders = preprocess_features(test_features)
    val_loaders = preprocess_features(val_features)
    # Training loop
    num_epochs = 100
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        train_loss = 0.0
        for pdb_id, loader in train_loaders.items():
            for inputs, labels in loader:
                optimizer.zero_grad()  # Zero the gradients
                outputs = model(inputs)  # Forward pass
                labels = labels.view(-1, 1)
                loss = criterion(outputs.squeeze(), labels.squeeze())  # Calculate the loss
                loss.backward()  # Backward pass
                optimizer.step()  # Update the weights
                train_loss += loss.item() * inputs.size(0)
            train_loss /= len(loader.dataset)

        # Print training statistics
        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}')
        train_losses.append(train_loss)

        # Validation
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        for pdb_id, loader in val_loaders.items():
            for inputs, labels in loader:
                outputs = model(inputs)  # Forward pass
                labels = labels.view(-1, 1)
                loss = criterion(outputs.squeeze(), labels.squeeze())  # Calculate the loss
                val_loss += loss.item() * inputs.size(0)
            val_loss /= len(loader.dataset)

        # Print validation statistics
        print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}')
        val_losses.append(val_loss)
    
    # Testing
    test_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    val_probs = []
    val_labels = []
    for pdb_id, loader in test_loaders.items():
        for inputs, labels in loader:
            outputs = model(inputs)  # Forward pass
            labels = labels.view(-1, 1)
            loss = criterion(outputs.squeeze(), labels.squeeze())  # Calculate the loss
            test_loss += loss.item() * inputs.size(0)

            # Calculate accuracy
            predicted = torch.round(outputs)  # Round the model outputs to get predictions
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            val_probs.extend(outputs.detach().numpy())
            val_labels.extend(labels.numpy())


        test_loss /= len(loader.dataset)
        
    fpr, tpr, thresholds = roc_curve(val_labels, val_probs)
    roc_auc = auc(fpr, tpr)
    # Print test statistics
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {correct_predictions / total_predictions * 100:.2f}%')
    print(f'ROC AUC: {roc_auc:.2f}')

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    # Save the model
    torch.save(model.state_dict(), 'model_100.pth')