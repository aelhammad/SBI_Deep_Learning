from setuptools import setup, find_packages

setup(
    name='deep-pocket',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'biopython==1.83',
        'numpy>=1.19.5,<2.0',  # Ensure NumPy version is compatible with scikit-learn
        'torch',
        'torchvision',
        'torchaudio',
        'pandas',
        'scikit-learn',  # No specific version specified, will use the latest compatible version with NumPy
        'matplotlib',
    ],
    python_requires='>=3.6',
    author='Allal El Hommad, Javier Herranz and Daniel Perez',
    author_email='allal.elhommad01@estudiant.upf.edu, javier.herranz01@estudiant.upf.edu, daniel.perez15@estudiant.upf.edu',
    description='Deep Pocket is a Python package that predicts binding pockets within protein structures using state-of-the-art deep learning techniques. By leveraging a neural network model trained on a diverse dataset of Protein Data Bank (PDB) files, Deep Pocket offers bioinformaticians and computational biologists a powerful tool for accurately identifying crucial binding sites. This package is designed to streamline drug discovery, protein function analysis, and molecular docking studies, making it an essential resource for researchers in structural biology and related fields.',
    long_description=open('../README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/aelhammad/deep-pocket',
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
