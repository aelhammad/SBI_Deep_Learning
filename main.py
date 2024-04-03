import torch
from sorted_pdb_second_data import MyModel,preprocess_features
from feature_extraction import ProteinFeatures
from Bio.PDB import PDBIO, Atom, Residue, Chain, Model, Structure

# Load the pre-trained model
input_size = 34
model = MyModel(input_size)
model.load_state_dict(torch.load('model.pth'))

# Ensure that the model is in evaluation mode
model.eval()

# Load and process your pdb data using preprocess_features function
pdb_file = 'dataset/182l.pdb'
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
binding_pocket_residues = [index for index, prediction in enumerate(predictions) if prediction > 0.1]
print(binding_pocket_residues)
# Filter the features of the binding pocket residues
filtered_features = [features[index] for index in binding_pocket_residues]


# Create a PDB file containing only the amino acids predicted to be in the binding pocket
structure = Structure.Structure(pdb_file)
model = Model.Model(0)
chain = Chain.Chain('A')
model.add(chain)
for index, feature in zip(binding_pocket_residues, filtered_features):
    residue = Residue.Residue((' ', index, ' '), 'ALA', index)
    chain.add(residue)
    atom = Atom.Atom('CA', (1.0, 2.0, 3.0), 1.0, 1.0, ' ', 'CA', index, 'C')
    residue.add(atom)

# Save the structure to a PDB file
io = PDBIO()
io.set_structure(structure)
io.save('binding_pocket_predicted.pdb')
