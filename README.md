This project consists of a deep learning model able to predict the binding pocket of a protein structure. Therefore it takes a PDB file as the input and returns a PDB file of the binding pocket and a list of the aminoacids that form that binding pocket


For this software to work it is required to have a working version of the dssp software (http://swift.cmbi.ru.nl/gv/dssp/). The recommended version is the 4.0.4 but any new version who runs the 'mkdssp' binary should work.


For MACOS users dssp can be installed with brew: 

brew install dssp

For Linux users dssp can be installed with apt: 

sudo apt-get install dssp

dssp can also be installed with conda, be aware that this may not be the case for computers with apple silicon (M1,M2 and M3 and so on) chips. 

conda install salilab::dssp
