import pickle
from sklearn.model_selection import train_test_split
import os
import glob
from feature_extraction import featurizer

def find_matched_pdb_files(pdb_dir, pocket_dir):
    '''
    Find matched PDB files and their corresponding pocket files.

    Args:
        pdb_dir (str): Path to the directory containing PDB files.
        pocket_dir (str): Path to the directory containing pocket PDB files.

    Returns:
        tuple: A tuple containing lists of matched PDB and pocket file paths.
    # Example usage:
    pdb_dir = '/path/to/pdb/directory'
    pocket_dir = '/path/to/pocket/directory'
    matched_pdb_files, matched_pocket_files = find_matched_pdb_files(pdb_dir, pocket_dir)
    print("Matched PDB files:", matched_pdb_files)
    print("Matched pocket files:", matched_pocket_files)
    '''
    # List all .pdb files in the PDB folder, remove the extension and the prefix
    pdb_files = [os.path.basename(f) for f in glob.glob(os.path.join(pdb_dir, '*.pdb'))]
    pdb_ids = [f[3:-4] for f in pdb_files if f.startswith('pdb') and f.endswith('.pdb')]
    # List all pocket files and remove the extension
    pocket_files = [os.path.basename(f) for f in glob.glob(os.path.join(pocket_dir, '*_pocket.pdb'))]
    pocket_ids = [f[:-11] for f in pocket_files if f.endswith('_pocket.pdb')]
    # Find the intersection of the two sets to ensure each PDB has its pocket
    matched_ids = set(pdb_ids).intersection(set(pocket_ids))
    # Construct the full filenames for the matched files
    matched_pdb_files = [os.path.join(pdb_dir, f'pdb{id}.pdb') for id in matched_ids]
    matched_pocket_files = [os.path.join(pocket_dir, f'{id}_pocket.pdb') for id in matched_ids]
    return matched_pdb_files, matched_pocket_files

def split_data(matched_files, test_size=0.2, val_size=0.25):
    '''
    Split data into training, validation, and test sets.

    Args:
        data (list): A list of data to be split.
        train_ratio (float, optional): Ratio of training data. Defaults to 0.8.
        val_ratio (float, optional): Ratio of validation data. Defaults to 0.1.
        test_ratio (float, optional): Ratio of test data. Defaults to 0.1.

    Returns:
        tuple: A tuple containing lists of training, validation, and test data.
    # Example usage:
    matched_files = list(zip(matched_pdb_files, matched_pocket_files))  # Assuming matched_pdb_files and matched_pocket_files are already obtained
    train_files, val_files, test_files = split_data(matched_files)
    print("Training files:", train_files)
    print("Validation files:", val_files)
    print("Test files:", test_files)
    '''
    # Split into training + validation and test sets
    train_val_files, test_files = train_test_split(matched_files, test_size=test_size, random_state=42)
    # Adjust the size of the validation set relative to the size of the training set
    adjusted_val_size = val_size / (1 - test_size)
    # Split the remaining data into training and validation sets
    train_files, val_files = train_test_split(train_val_files, test_size=adjusted_val_size, random_state=42)
    return train_files, val_files, test_files

# Saving with Pickle
def save_features_pickle(data, filepath):
    '''
    Save features dictionary as a pickle file.

    Args:
        features_dict (dict): Dictionary containing features.
        filename (str): Name of the pickle file to save.
    # Example usage:
    data_to_save = {'example': 'data'}
    save_features_pickle(data_to_save, 'example_data.pkl')
    '''
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

# Loading with Pickle
def load_features_pickle(filepath):
    '''
    Process PDB and pocket PDB files, extract features, and save them as pickle files.

    Args:
        pdb_dir (str): Path to the directory containing PDB files.
        pocket_dir (str): Path to the directory containing pocket PDB files.
    # Example usage
    loaded_data = load_features_pickle('example_data.pkl')
    print("Loaded data:", loaded_data)
    '''
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def process_data(pdb_dir, pocket_dir):
    '''
    Process PDB and pocket PDB files, extract features, and save them as pickle files.

    Args:
        pdb_dir (str): Path to the directory containing PDB files.
        pocket_dir (str): Path to the directory containing pocket PDB files.
    # Example usage
    pdb_dir = '/path/to/pdb/directory'
    pocket_dir = '/path/to/pocket/directory'
    process_data(pdb_dir, pocket_dir)
    '''
    matched_pdb_files_list, matched_pocket_files_list = find_matched_pdb_files(pdb_dir, pocket_dir)
    matched_files = list(zip(matched_pdb_files_list, matched_pocket_files_list))[:1000]
    train_files, val_files, test_files = split_data(matched_files)

    features_dict_final_train = featurizer(train_files)
    save_features_pickle(features_dict_final_train, 'train_features_final.pkl')
    print("Train features saved.")

    features_dict_final_val = featurizer(val_files)
    save_features_pickle(features_dict_final_val, 'val_features_final.pkl')
    print("Validation features saved.")

    features_dict_final_test = featurizer(test_files)
    save_features_pickle(features_dict_final_test, 'test_features_final.pkl')
    print("Test features saved.")