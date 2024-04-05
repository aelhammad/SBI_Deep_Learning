Deep Pocket is a Python package designed to predict the binding pocket of a protein structure provided by the user. Leveraging deep learning techniques, the package utilizes a neural network model trained on a dataset comprising 1000 Protein Data Bank (PDB) files. With its focus on accuracy and efficiency, Deep Pocket aims to assist bioinformaticians in identifying binding pockets within protein structures, aiding in drug discovery, protein function analysis, and molecular docking studies.

Deep Pocket is primarily aimed at bioinformaticians, computational biologists, and researchers working in the field of structural biology. It provides a valuable tool for analyzing protein structures and identifying potential binding sites, facilitating drug discovery, protein-protein interaction studies, and other molecular biology applications.

Whether you are conducting research in academia, working in the pharmaceutical industry, or exploring protein structure-function relationships, Deep Pocket offers a robust and efficient solution for predicting binding pockets with high accuracy and reliability.

Main Features:
- Predict Binding Pockets: Utilize the trained neural network model to predict the binding pockets of a protein structure.
- Bioinformatic Applications: Targeted towards bioinformaticians and researchers in the life sciences for various computational biology tasks.
- Deep Learning Techniques: Harnesses the power of deep learning to accurately identify binding sites within protein structures.
- Ease of Use: Designed with user-friendliness in mind, allowing for seamless integration into existing workflows.

For this software to work it is required to have a working version of the dssp software (http://swift.cmbi.ru.nl/gv/dssp/). The recommended version is the 4.0.4 but any new version who runs the 'mkdssp' binary should work.

For MACOS users dssp can be installed with brew: 

brew install dssp

For Linux users dssp can be installed with apt: 

sudo apt-get install dssp

dssp can also be installed with conda, be aware that this may not be the case for computers with apple silicon (M1,M2 and M3 and so on) chips. 

conda install salilab::dssp


