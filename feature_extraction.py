from Bio.PDB import PDBParser, NeighborSearch, Selection, is_aa 
import pandas as pd 
import subprocess
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import os
import torch
import torch.nn as nn
import os
import numpy as np
from Bio.PDB.DSSP import DSSP

# Set the working directory
os.chdir('/Users/allalelhommad/PYT/SBPYT_project')

class ProteinFeatures:
    
    def __init__(self, pdb_file, pocket_pdb_file=None):
        self.dssp_executable = '/opt/homebrew/bin/mkdssp' 
        self.pocket_pdb_file = pocket_pdb_file
        self.pdb_file = pdb_file
        self.parser = PDBParser(QUIET=True)
        self.amino_acids = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU', 
                            'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR']
        self.secondary_structure_codes = ['H', 'B', 'E', 'G', 'I', 'T', 'S', ' ', '.']
        self.pocket_residues = self.load_pocket_residues()
        torch.manual_seed(42)  # Set a fixed seed for random number generation
        self.structure = self.parser.get_structure('protein', self.pdb_file)
        self.encoding_ss, self.encoding_aa = self.one_hot_encoding() 

    
    def extract_sequence(self):
        parser = PDBParser()
        structure = parser.get_structure('protein', self.pdb_file)
        model = structure[0]  # Assuming single model
        chain = model['A']  # Assuming chain A, change as needed
        residues = chain.get_residues()

        aminoacid_counts = {}
        total_residues = 0

        for residue in residues:
            if residue.get_id()[0] == ' ':
                aminoacid = residue.get_resname()
                aminoacid_counts[aminoacid] = aminoacid_counts.get(aminoacid, 0) + 1
                total_residues += 1

        aminoacid_frequencies = {aminoacid: count / total_residues for aminoacid, count in aminoacid_counts.items()}

        return aminoacid_frequencies
    
    def calculate_total_contact(self):
        """Calculate the total contact for each residue based on a distance threshold."""
        atom_list = Selection.unfold_entities(self.structure, 'A')  # Use self.structure
        ns = NeighborSearch(atom_list)
        total_contact_dict = {}

        for chain in self.structure.get_chains():  # Use self.structure
            for residue in chain:
                if not is_aa(residue, standard=True):  # Skip non-amino acid entities, ensure to import is_aa
                    continue
                residue_key = (chain.id, residue.resname)
                try:
                    alpha_carbon = residue['CA']  # Assuming the alpha carbon is labeled as 'CA'
                    contacts = ns.search(alpha_carbon.get_coord(), 5.0, level='A')  # Search within 10Å radius
                    total_contact_dict[residue_key] = len(contacts) - 1  # Subtract 1 to exclude the residue itself
                except KeyError:
                    total_contact_dict[residue_key] = 0  # Set total contact to 0 if alpha carbon is missing

        return total_contact_dict



    
    def extract_secondary_structure(self):
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', self.pdb_file)
        model = structure[0]

        # Generate DSSP data using Biopython's DSSP wrapper
        dssp = DSSP(model, self.pdb_file, dssp=self.dssp_executable)

        # Structured data to store the features
        structured_results = {}

        # Iterate over DSSP output to populate structured_results
        for key in dssp.keys():
            res_id = (key[0], key[1][1])  # Chain ID and residue number (ignoring insertion code for simplicity)
            _, _, ss, access, phi, psi = dssp[key][:6]  # Extract the necessary data
            structured_results[res_id] = {
                'secondary_structure': ss,
                'solvent_accessibility': access,
                'phi': phi,
                'psi': psi
            }

        return structured_results
    
    def load_pocket_residues(self):
        pocket_residues_set = set()
        if self.pocket_pdb_file:
            with open(self.pocket_pdb_file, 'r') as file:
                for line in file:
                    if line.startswith('ATOM'):
                        chain = line[21]
                        residue = line[22:26].strip()
                        pocket_residues_set.add((chain, residue))
        return pocket_residues_set
    
    def one_hot_encoding(self):
        encoding_ss = {}
        encoding_aa = {}
        
        # Encode secondary structure
        for i, code in enumerate(self.secondary_structure_codes):
            one_hot_vector_ss = torch.zeros(len(self.secondary_structure_codes))
            one_hot_vector_ss[i] = 1
            encoding_ss[code] = one_hot_vector_ss
        
        # Encode amino acids
        for i, amino_acid in enumerate(self.amino_acids):
            one_hot_vector_aa = torch.zeros(len(self.amino_acids))
            one_hot_vector_aa[i] = 1 
            encoding_aa[amino_acid] = one_hot_vector_aa
        
        return encoding_ss, encoding_aa


    def extract_features(self):
        structure = self.parser.get_structure('protein', self.pdb_file)
        model = structure[0]  # Assuming a single model for simplicity
        dssp_data = self.extract_secondary_structure()
        all_features = []
        for chain in model.get_chains():
            for residue in chain.get_residues():
                if residue.get_id()[0] == ' ':  # Only standard residues
                    residue_id = residue.get_id()
                    pdb_id = self.pdb_file.split('/')[-1].split('.')[0]
                    residue_name = residue.get_resname()
                    secondary_structure = dssp_data.get((chain.id, residue_id[1]), {}).get('secondary_structure', ' ')
                    solvent_accessibility = dssp_data.get((chain.id, residue_id[1]), {}).get('solvent_accessibility', 0)
                    psi_angle = dssp_data.get((chain.id, residue_id[1]), {}).get('psi', 0)
                    phi_angle = dssp_data.get((chain.id, residue_id[1]), {}).get('phi', 0)
                    In_pocket = int((chain.id, str(residue_id[1])) in self.pocket_residues)
                    total_contact = self.calculate_total_contact().get((chain.id, residue.resname), 0)
                    secondary_structure_one_hot = [self.encoding_ss[code] for code in secondary_structure]
                    amino_acid_one_hot = self.encoding_aa[residue_name]

                    feature_dict = {
                        'PDB_ID': f'{chain.id}_{residue_id[1]}_{residue_name}',
                        'PDB':pdb_id,
                        'Residue_Name': amino_acid_one_hot,
                        'In_Pocket': In_pocket,
                        'Secondary_structure': secondary_structure_one_hot,
                        'Solvent_accesibility': float(solvent_accessibility),
                        'Psi_angle': psi_angle,
                        'Phi_angle': phi_angle,
                        'Total_contact': total_contact,
                    }
                    
                    all_features.append(feature_dict)
                    
        return all_features

pdbnames=['1uho','1x8v','182l','1upv','3ies','3zsx','3zvu','4m12','4m13']
pdb_files=[f'dataset/{pdbname}.pdb' for pdbname in pdbnames]
pocket_pdb_files=[f'dataset/pocket_pdb/{pdbname}_pocket.pdb' for pdbname in pdbnames]

def features(pdb_files, pocket_pdb_files):
    all_features = []  # Initialize an empty list to accumulate features for all PDB files
    for pdb_file, pocket_pdb_file in zip(pdb_files, pocket_pdb_files):
        pf = ProteinFeatures(pdb_file, pocket_pdb_file)
        features = pf.extract_features()
        all_features.extend(features)  # Append features for the current PDB file to the list
    return all_features


df = pd.DataFrame(features(pdb_files, pocket_pdb_files))
df.to_csv('tentest.csv', index=False)