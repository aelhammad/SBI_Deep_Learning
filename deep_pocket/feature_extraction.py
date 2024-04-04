# Import necessary libraries
from Bio.PDB.PDBList import PDBList
from Bio.PDB.DSSP import DSSP
import torch
from Bio.PDB import PDBParser, NeighborSearch, Selection, is_aa 
import pandas as pd 
import os
import numpy as np

# Define the ProteinFeatures class
class ProteinFeatures:
    '''
    Class used to extract features from a protein structure.

    Attributes:
        pdb_file (str): Path to the PDB file containing the protein structure.
        pocket_pdb_file (str, optional): Path to the pocket PDB file.
    # Example usage
    pdb_file = '/path/to/pdb/file.pdb'
    pocket_pdb_file = '/path/to/pocket/pdb/file.pdb'
    pf = ProteinFeatures(pdb_file, pocket_pdb_file)
    features = pf.extract_features()
    print(features)
    '''
    
    def __init__(self, pdb_file, pocket_pdb_file=None):
        # Set the path to the DSSP executable
        self.dssp_executable = './mkdssp' 
        self.pocket_pdb_file = pocket_pdb_file
        self.pdb_file = pdb_file
        self.parser = PDBParser(QUIET=True)
        self.amino_acids = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU', 
                            'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR']
        self.secondary_structure_codes = ['H', 'B', 'E', 'G', 'I', 'T', 'S', ' ', '-', 'P']
        self.pocket_residues = self.load_pocket_residues()
        torch.manual_seed(42)  # Set a fixed seed for random number generation
        self.structure = self.parser.get_structure('protein', self.pdb_file)
        self.encoding_ss, self.encoding_aa = self.one_hot_encoding() 
    
    def extract_sequence(self):
        '''
        Extract the amino acid sequence from the PDB file.

        Returns:
            dict: A dictionary containing the frequencies of each amino acid in the sequence.
        # Example usage
        pf = ProteinFeatures(pdb_file)
        sequence = pf.extract_sequence()
        print(sequence)  
        '''
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
        '''
        Calculate the total contact for each residue based on a distance threshold.

        Returns:
            dict: A dictionary containing the total contact for each residue.
        # Example usage
        pf = ProteinFeatures(pdb_file)
        total_contact = pf.calculate_total_contact()
        print(total_contact)
        # Example usage
        pf = ProteinFeatures(pdb_file)
        total_contact = pf.calculate_total_contact()
        print(total_contact)
        '''
        atom_list = Selection.unfold_entities(self.structure, 'A')
        ns = NeighborSearch(atom_list)
        total_contact_dict = {}
        for chain in self.structure.get_chains():  
            for residue in chain:
                if not is_aa(residue, standard=True):
                    continue
                residue_key = (chain.id, residue.resname)
                try:
                    alpha_carbon = residue['CA'] 
                    contacts = ns.search(alpha_carbon.get_coord(), 5.0, level='A') 
                    total_contact_dict[residue_key] = len(contacts) - 1  
                except KeyError:
                    total_contact_dict[residue_key] = 0 
        return total_contact_dict

    def extract_secondary_structure(self):
        '''
        Extract secondary structure and solvent accessibility using DSSP.

        Returns:
            dict: A dictionary containing secondary structure and solvent accessibility information for each residue.
        # Example usage
        pf = ProteinFeatures(pdb_file)
        secondary_structure = pf.extract_secondary_structure()
        print(secondary_structure)
        '''
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', self.pdb_file)
        model = structure[0]
        dssp = DSSP(model, self.pdb_file, dssp=self.dssp_executable)
        structured_results = {}
        for key in dssp.keys():
            res_id = (key[0], key[1][1])  
            _, _, ss, access, phi, psi = dssp[key][:6]
            structured_results[res_id] = {
                'secondary_structure': ss,
                'solvent_accessibility': access,
                'phi': phi,
                'psi': psi
            }
        return structured_results
    
    def load_pocket_residues(self):
        '''
        Load the pocket residues from a pocket PDB file.

        Returns:
            set: A set containing tuples of chain and residue IDs representing pocket residues.
        # Example usage
        pf = ProteinFeatures(pdb_file, pocket_pdb_file)
        pocket_residues = pf.load_pocket_residues()
        print(pocket_residues)
        '''
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
        '''
        Create one-hot encoding for amino acids and secondary structure.

        Returns:
            tuple: A tuple containing dictionaries of one-hot encodings for amino acids and secondary structure.
        # Example usage
        pf = ProteinFeatures(pdb_file)
        encoding_ss, encoding_aa = pf.one_hot_encoding()
        print(encoding_ss)
        print(encoding_aa)
        '''
        encoding_ss = {}
        encoding_aa = {}
        for i, code in enumerate(self.secondary_structure_codes):
            one_hot_vector_ss = torch.zeros(len(self.secondary_structure_codes))
            one_hot_vector_ss[i] = 1
            encoding_ss[code] = one_hot_vector_ss
        for i, amino_acid in enumerate(self.amino_acids):
            one_hot_vector_aa = torch.zeros(len(self.amino_acids))
            one_hot_vector_aa[i] = 1 
            encoding_aa[amino_acid] = one_hot_vector_aa
        return encoding_ss, encoding_aa
    
    def extract_features(self):
        '''
        Extract features from the protein structure.

        Returns:
            list: A list of dictionaries containing features for each residue.
        # Example usage
        pf = ProteinFeatures(pdb_file, pocket_pdb_file)
        features = pf.extract_features()
        print(features)
        '''
        structure = self.parser.get_structure('protein', self.pdb_file)
        model = structure[0]
        dssp_data = self.extract_secondary_structure()
        all_features = []
        for chain in model.get_chains():
            for residue in chain.get_residues():
                if residue.get_id()[0] == ' ':  
                    residue_id = residue.get_id()
                    pdb_id = self.pdb_file.split('/')[-1].split('.')[0]
                    residue_name = residue.get_resname()
                    secondary_structure = dssp_data.get((chain.id, residue_id[1]), {}).get('secondary_structure', ' ')
                    solvent_accessibility = dssp_data.get((chain.id, residue_id[1]), {}).get('solvent_accessibility', 0)
                    solvent_accessibility = float(solvent_accessibility) if solvent_accessibility else 0
                    psi_angle = dssp_data.get((chain.id, residue_id[1]), {}).get('psi', np.nan)
                    phi_angle = dssp_data.get((chain.id, residue_id[1]), {}).get('phi', np.nan)
                    In_pocket = int((chain.id, str(residue_id[1])) in self.pocket_residues)
                    total_contact = self.calculate_total_contact().get((chain.id, residue.resname), 0)
                    amino_acid_one_hot = self.encoding_aa[residue_name]
                    secondary_structure_one_hot = [self.encoding_ss[code] for code in secondary_structure]
                    feature_dict = {
                        'PDB_ID': pdb_id,
                        'Residue_Name': amino_acid_one_hot,
                        'In_Pocket': In_pocket,
                        'Secondary_structure': secondary_structure_one_hot,
                        'Solvent_accessibility': solvent_accessibility,
                        'Psi_angle': psi_angle,
                        'Phi_angle': phi_angle,
                        'Total_contact': total_contact
                    }
                    all_features.append(feature_dict)
        return all_features

def download_pdb_files(pocket_pdb_directory, pdb_directory):
    '''
    Download PDB files from the given pocket PDB directory and save them in the specified PDB directory.

    Parameters:
    - pocket_pdb_directory (str): Path to the directory containing the pocket PDB files.
    - pdb_directory (str): Path to the directory where the downloaded PDB files will be saved.
    # Example usage for download_pdb_files
    pocket_pdb_directory = '/path/to/pocket/pdb/directory'
    pdb_directory = '/path/to/pdb/directory'
    downloaded_pdb_files = download_pdb_files(pocket_pdb_directory, pdb_directory)
    print("Downloaded PDB files:", downloaded_pdb_files)
    '''
    # Initialize PDBList
    pdbl = PDBList()
    # Create the PDB directory if it doesn't exist
    if not os.path.exists(pdb_directory):
        os.makedirs(pdb_directory)
    # List to hold the formatted PDB names
    pdb_names = []
    # Loop through each file in the directory
    for filename in os.listdir(pocket_pdb_directory):
        if filename.endswith("_pocket.pdb"):
            # Extract the PDB identifier from the filename
            pdb_id = filename.split("_pocket.pdb")[0]
            # Append the formatted name to the list
            pdb_names.append(pdb_id.upper())  # PDB IDs are typically uppercase
    for pdb_id in pdb_names:
        # Retrieve PDB file
        pdbl.retrieve_pdb_file(pdb_id, pdir=pdb_directory, file_format='pdb')

        print(f"Downloaded: {pdb_id}")
    return pdb_names

def featurizer(test_matched_files):
    '''
    Extract features from a list of PDB files and return a list of dictionaries. 
    To train the model you need to process (process_data.py) the data before use featurizer.
    To process the data: find_matched_pdb_files -> split_data -> save_features_pickle
    Features calculated:
                        'PDB_ID'
                        'Residue_Name'
                        'In_Pocket'
                        'Secondary_structure'
                        'Solvent_accessibility'
                        'Psi_angle'
                        'Phi_angle'
                        'Total_contact'
    Args:
        test_matched_files (list): A list of tuples containing paths to PDB files and their corresponding pocket PDB files.

    Returns:
        list: A list of dictionaries containing features for each residue.
    # Example usage for featurizer
    test_matched_files = [
        ('/path/to/pdb/file1.pdb', '/path/to/pocket/pdb/file1_pocket.pdb'),
        ('/path/to/pdb/file2.pdb', '/path/to/pocket/pdb/file2_pocket.pdb')
    ]
    features = featurizer(test_matched_files)
    print("Extracted features:", features)
    '''
    all_features_list = []
    for pdb_file, pocket_pdb_file in test_matched_files:
        pf = ProteinFeatures(pdb_file, pocket_pdb_file)
        features_list = pf.extract_features()  # This is a list of dictionaries
        all_features_list.extend(features_list)  # Append features for the current PDB file to the list
    return all_features_list
