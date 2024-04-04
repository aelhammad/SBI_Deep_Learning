import torch
from model import MyModel, preprocess_features
from feature_extraction import ProteinFeatures
from Bio.PDB import PDBIO, Atom, Residue, Chain, Model, Structure
import os
from Bio.PDB import PDBParser
import os


# Load the pre-trained model
input_size = 34
model = MyModel(input_size)
model.load_state_dict(torch.load('models/model_100.pth'))

# Ensure that the model is in evaluation mode
model.eval()

# Load and process your pdb data using preprocess_features function
pdb_list = []
pdb_file = 'example_pdb/2ay2.pdb'
# Crear un analizador PDB
parser = PDBParser()

# Parsear la estructura del PDB
structure = parser.get_structure("structure", pdb_file)
pdb_list = [pdb_file]
pf = ProteinFeatures(pdb_file)
features = pf.extract_features()
pdb_loader = preprocess_features(features)
print(pdb_loader)

predictions = []
with torch.no_grad():
    for pdb_id, loader in pdb_loader.items():
        # Check if there's only one batch in the loader
        if len(loader) == 1:
            # Extract the single batch
            inputs, _ = next(iter(loader))
            inputs = torch.nan_to_num(inputs, nan=inputs.mean())
        else: #change this.
            # If there are multiple batches, concatenate them into one batch
            inputs, _ = torch.cat([batch[0] for batch in loader]), None
            inputs = torch.nan_to_num(inputs, nan=inputs.mean())

        print(inputs)
        outputs = model(inputs)
        print(outputs)
        predictions.extend(outputs.tolist())

# Filter the predicted binding pocket residues
predictions = [item for sublist in predictions for item in sublist]
binding_pocket_residues = [index for index, prediction in enumerate(predictions) if prediction > 0.8]
print(binding_pocket_residues)
# Parsear la estructura del PDB
structure = parser.get_structure("structure", pdb_file)

# Obtener el número máximo de residuos para cada cadena
max_residues = {}
for model in structure:
    for chain in model:
        residues = [res.get_id()[1] for res in chain.get_residues() if res.get_id()[0] == " "]
        max_residues[chain.get_id()] = max(residues)

max_residue_A = max_residues['A']
# Lista para almacenar las duplas (número, cadena)
residue_chain_tuples = []

# Iterar sobre los residuos en la lista binding_pocket_residues
for residue in binding_pocket_residues:
    if residue <= max_residue_A:
        # Residuo pertenece a la cadena A
        residue_chain_tuples.append((residue, 'A'))
    else:
        # Residuo pertenece a la cadena B, restar el máximo de la cadena A
        residue_chain_tuples.append((residue - max_residue_A, 'B'))

# Imprimir las duplas (número, cadena)
print(residue_chain_tuples)


# Crear una lista para almacenar las líneas del pocket de unión
predicted_binding_pocket_lines = []

# Iterar sobre las duplas (número, cadena)
# Abrir el archivo PDB
with open(pdb_file, 'r') as pdb_input:
    # Iterar sobre las líneas del archivo PDB
    for line in pdb_input:
        if line.startswith('ATOM'):
        # Extraer el número de residuo y la cadena de la línea actual
            residue_number = int(line[22:26].strip())
            chain_id = line[21]
            for residue, chain in residue_chain_tuples:
                if residue == residue_number and chain == chain_id:
                    predicted_binding_pocket_lines.append(line)
print(predicted_binding_pocket_lines)

pdb_file_name = os.path.basename(pdb_file)
output_pdb_file = f'predicted_binding_pocket_{pdb_file_name}'
with open(output_pdb_file, 'w') as output_pdb:
    output_pdb.write(''.join(predicted_binding_pocket_lines))

print(f"Predicted binding pocket residues saved to {output_pdb_file}")
