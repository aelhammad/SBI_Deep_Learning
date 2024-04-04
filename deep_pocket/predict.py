import torch
import sys
import time  # Import time module
from deep_pocket.model import MyModel, preprocess_features
from feature_extraction_allal import ProteinFeatures
from Bio.PDB import PDBParser
import os
import argparse
from Bio.PDB.PDBExceptions import PDBConstructionException

def predict_pdb(pdb_file_path, verbose=False):
    '''
    Predict binding pocket residues in a given PDB file using a pre-trained model.

    Parameters:
        pdb_file_path (str): Path to the input PDB file.
        verbose (bool): Flag to enable verbose mode.

    Returns:
        None
    '''
    try:
        if not pdb_file_path.lower().endswith('.pdb'):
            raise ValueError("Error: Input file must have a .pdb extension")

        # Check if the trust_level is within the valid range
        if args.trust_level > 0.99:
            raise ValueError("Error: Trust level must not exceed 0.99")
        
        print("Performing prediction", end="", flush=True)
        for _ in range(3):  # Print three dots
            print(".", end="", flush=True)
            time.sleep(1)  # Wait for one second between dots
        print("\n")
        # Redirect stdout and stderr to suppress output and warnings
        if not verbose:
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')

        # Load the pre-trained model
        input_size = 34
        model = MyModel(input_size)
        model.load_state_dict(torch.load('models/model_100.pth'))
        model.eval()

        # Load and process the PDB data using preprocess_features function
        parser = PDBParser()
        structure = parser.get_structure("structure", pdb_file_path)
        pf = ProteinFeatures(pdb_file_path)
        features = pf.extract_features()
        pdb_loader = preprocess_features(features)

        predictions = []
        with torch.no_grad():
            for pdb_id, loader in pdb_loader.items():
                if len(loader) == 1:
                    inputs, _ = next(iter(loader))
                    inputs = torch.nan_to_num(inputs, nan=inputs.mean())
                else:
                    inputs, _ = torch.cat([batch[0] for batch in loader]), None
                    inputs = torch.nan_to_num(inputs, nan=inputs.mean())

                outputs = model(inputs)
                predictions.extend(outputs.tolist())

        # Filter the predicted binding pocket residues
        predictions = [item for sublist in predictions for item in sublist]
        binding_pocket_residues = [index for index, prediction in enumerate(predictions) if prediction > args.trust_level]

        # Get the maximum residue number for each chain
        max_residues = {}
        for model in structure:
            for chain in model:
                residues = [res.get_id()[1] for res in chain.get_residues() if res.get_id()[0] == " "]
                max_residues[chain.get_id()] = max(residues)

        max_residue_A = max_residues.get('A', 0)
        residue_chain_tuples = []

        # Generate residue-chain tuples
        for residue in binding_pocket_residues:
            if residue <= max_residue_A:
                residue_chain_tuples.append((residue, 'A'))
            else:
                residue_chain_tuples.append((residue - max_residue_A, 'B'))

        predicted_binding_pocket_lines = []

        output_residues = [] 
        with open(pdb_file_path, 'r') as pdb_input: # Read the input PDB file
            for line in pdb_input:
                if line.startswith('ATOM'):
                    residue_number = int(line[22:26].strip())
                    chain_id = line[21]
                    residue_name = line[17:20].strip()  # Extract the residue name
                    for residue, chain in residue_chain_tuples:
                        if residue == residue_number and chain == chain_id:
                            predicted_binding_pocket_lines.append(line)
                            output_residues.append((residue_name, residue, chain))
        output_residues = set(output_residues)
        output_residues = list(output_residues)
        output_residues.sort(key=lambda x: (x[2], x[1]))

        # Restore stdout and stderr
        if not verbose:
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

        # Determine the output directory and save the processed PDB file
        output_dir = os.path.dirname(pdb_file_path)
        pdb_file_name = os.path.basename(pdb_file_path)
        output_pdb_file = os.path.join(output_dir, f'predicted_binding_pocket_{pdb_file_name}')
        output_txt_file = os.path.join(output_dir, f'{pdb_file_name}_predicted_pocket.txt')

        with open(output_txt_file, 'w') as output_txt: # Save the predicted binding pocket residues to a text file
            for residue_name, residue_number, chain in output_residues:
                output_txt.write(f"{residue_name} {residue_number} {chain}\n")

        with open(output_pdb_file, 'w') as output_pdb: # Save the predicted binding pocket residues to a PDB file
            output_pdb.write(''.join(predicted_binding_pocket_lines))
        
        # Print the output_residues to standard output
        print("\nResidue Name  Residue Number  Chain")
        for residue_name, residue_number, chain in output_residues:
            print(f"{residue_name:<13} {residue_number:<15} {chain}")
        print(f"Predicted binding pocket residues saved to {output_txt_file}")
        print(f"Predicted binding pocket PDB saved to {output_pdb_file}")

    except FileNotFoundError: # Handle file not found error
        print("Error: PDB file not found.")
    except PDBConstructionException as e:
        print(f"Error: Unable to parse the PDB file: {e}")
    except Exception as e: # Handle other exceptions
        print(f"Error: An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict binding pocket residues in a given PDB file.') # Create an ArgumentParser object
    parser.add_argument('--trust_level', type=float, default=0.7, help='Trust level for the prediction')
    parser.add_argument('pdb_file_path', type=str, help='Path to the input PDB file')
    parser.add_argument('--verbose','-v', action='store_true', help='Enable verbose mode')

    args = parser.parse_args()
    predict_pdb(args.pdb_file_path, args.verbose)
