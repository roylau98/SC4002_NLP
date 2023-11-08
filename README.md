# SC4002_NLP

### convert this to a READMD.txt later  

## Setup instructions

The following instructions setups a new virtual environment for python and installs the needed libraries.  
These instructions assumes that the user is using a Windows machine. Ensure that Python 3.11 is installed as well.

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Ensure that 2 directories are created in the root folder. For "CoNLL2003_dataset/", put "eng.testa", "eng.testb" and "eng.train" into it 
and for "TREC_dataset/" include the "test.csv" and "train.csv" from Kaggle.

If the host machine that is running the code has a GPU with cuda you can install torch with cuda support instead 
using the command given in the PyTorch documentation https://pytorch.org/get-started/locally/

## Files 

The codes are given in the form of Jupyter notebooks. Files prepended with Q1_ contains code for Part 1 (Sequence Tagging: NER), while 
files prepended with Q2_ contains code for Part 2 (Sentence-Level Categorization: Question Classification).

Before running any of the Q2 models, one should run "Q2_preprocessing.ipynb" which will generate the updated dataset which will be used by the models. 
There are no restrictions on which files can be run first for Q1_ notebooks.

## Running the code

## Explanation of sample output

## Done by
Poh Yang Quan Eugene  
Wong Yi Pun  
Jeremy U Keat  
Ng Yue Jie Alphaeus  
Roy Lau Run-Xuan  

## Acknowledgements
CoNLL2023: https://github.com/TheAnig/NER-LSTM-CNN-Pytorch/tree/master/data  
TREC: https://www.kaggle.com/datasets/thedevastator/the-trec-question-classification-dataset-a-longi/