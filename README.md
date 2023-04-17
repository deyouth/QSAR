# 4D-QSAR modeling program

This program base on LQTA-QSAR method.It can be used in the LINUX operating system.

For LQTA-QSAR, please cite:

>J.P.A. Martins, E.G. Barbosa, K.F.M. Pasqualoto, M.M.C. Ferreira. J. Chem. Inf. Model. 2009, 49, 1428–1436.

>E.G. Barbosa, M.M.C. Ferreira, Mol. Inf. 2012, 31, 75–84.
>Tian Lu, Feiwu Chen, J. Comput. Chem., 33, 580-592 (2012)
### Prepare the following 4 (or 5) *.txt files：
#### names.txt
It contains all names of compounds,The text file will be:

#### index.txt
Each line of this file must contain the atom indexes from Gaussian that will be aligned, separeted by "tab". For example, the Gaussian atoms indexes of m01 and m02 that will be aligned are 1,3,5,7. For m03 these atoms have different Gaussian indexes, that are 2,4,5,9. The "index.txt" file will be:

>1 3  5	7

#### act.txt
It contains the bioactive value (pIC<sub>50</sub> value) of all compounds, corresponding to the name.txt. The text file will be:

#### ref.txt
It contains the name of the reference compound used in the alignment step. For example, m02 was chosen as the alignment reference. The text file will be:
#### test.txt (Optional)

Manually specify which compounds are used as test sets. If there is no such file, the program will randomly pick one-fifth of the compounds as a test set. For example, m02 is in the test set and m01,m03 are in the training set, the text file will be:

1.Firstly, use Gaussian for structural optimization, which is not limited to the base set and requires the addition of charge analysis and iop advanced instructions to output charge files, such as:
#opt freq b3lyp/cc-pvdz empiricaldispersion=gd3bj pop=mk iop(6/33=2,6/42=6,6/50=1)

2.“g09_input” is used to batch modify Gaussian input files. How to use: Put all the *.gjf files saved by GaussView in the same folder, create a new ref.txt file, and write the used memory, thread number, method, and task name line by line in it, for example:
6                      # use 6GB memory
5                      #Use 5 threads to calculate
opt freq b3lyp/6-311g  #method
top                    # task name

3.$g09_input This script will automatically help the user with the preparation of all Gaussian input files. If there is no ref.txt file in the folder, a corresponding prompt will be printed out. g09_run is used to submit Gaussian computing tasks sequentially in batches, put all *.gjf input files ready to run into the same folder, open the console in this folder, and enter:
>$g09_run

4.RESP2_ Noopt.sh and Multiwfn_ 3.8 was used to fit RESP charges, and sobtob was used to generate small molecule topology files.Please refer to the following website for details：http://sobereva.com/476

5.Run "b4_run_dyn.py" to delete the atom number in the second column in the *.gro file, and then generate a posre position restriction file *.itp for each compound according to the number of atoms in the file. Run "run_dyn_m", create a folder for each compound, and put the compound-related files into the corresponding folder.Then "run din_script_m" to start the simulation, the script contains instructions to call GROMACS for dynamic simulation.

6.Run "mk_ ndx_ m" Create an ndx folder and change the operation directory to that folder. Generate a *. ndx file for each compound and write the index number of the aligned atoms of the compound molecule according to index.txt. Afterwards, run "mk_ grid_ inps_ m" Create grid_ Inps folder and generate a folder for each compound. Use the gmx trjconv tool of GROMACS to read and stack the CEP trajectory files for each compound

7.Run "mkff", create a ffcargasnb.itp file, and write all the atomic types and parameters contained in ff * nb.itp to it, with each atomic type written only once.Run "mkbox_ m" Record the maximum and minimum X, Y, and Z coordinate values of all compound structures in a box.txt for calculating the size of the virtual box. Run "mklist_ m" Create a list.txt and write the file names of all files in that folder. Copy the user-prepared names.txt and act.txt to this folder. Run "LQTA.py" to calculate the interaction energy between the probe and the compound structure

8. Finally, use "Detective.py" for data processing and model construction

# MIA-QSAR
MIA-QSAR modeling program
## Requirements
python >= 3
## Installing MIA-QSAR
MIA-QSAR no need to install,put the program in any directory you want.

Before your first use, open terminal and type:

$ python -m pip install --upgrade pip

$ pip install --upgrade numpy sklearn matplotlib pillow

## How to use MIA-QSAR
#### Draw 2D molecular structures of compounds,and each one save as a picture.The structure cannot be twisted and stretched. Font size, bond length and bond angle should be consistent. Pictures are best named with numbers. You can use any chemical drawing software you want (ChemBioDraw is recommended).Put all pictures in a folder and prepare following text file:
#### 1.act.txt

It contains the bioactive value (pIC<sub>50</sub> value) of all compounds, corresponding to the name of pictures . For example, the text file will be:

>3.2

#### 2. test.txt (Optional)
Manually specify which compounds are used as test sets. If there is no such file, the program will randomly pick one-fifth of the compounds as a test set. For example, 12,24,27 and 32 are in the test set, the text file will be:

>12

#### After that, copy ImageKiller.py and ImageDetective.py to this folder. Run the ImageKiller.py to align structures and generate descriptors. Run ImageDetective.py to build the model.
