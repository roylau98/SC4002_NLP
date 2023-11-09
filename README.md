# SC4002_NLP

### convert this to a READMD.txt later  

## Setup instructions

The following instructions setups a new virtual environment for python and installs the needed libraries.  
These instructions assumes that the user is using a Windows machine. Ensure that Python 3.11 is installed as well, do not use Python 3.12. 

```cmd
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Ensure that 2 directories are created in the root folder. For "CoNLL2003_dataset/", put "eng.testa", "eng.testb" and "eng.train" into it 
and for "TREC_dataset/" include the "test.csv" and "train.csv" from Kaggle. 

If the machine that is running the code has a GPU with cuda you can install torch with cuda support instead 
using the command given in the PyTorch documentation https://pytorch.org/get-started/locally/

## Folder structure and files 

The following is a tree structure that shows the files found in this folder SC4002_G3. A brief description of what each file is for is included at the side.

```markdown
SC4002_G3
├── Assignment_NLP_SC4002.pdf           # Assignment instructions
├── f1_score_vs_epoch.png               # Plot of f1 score in each epoch for part 1
├── Q1_part1.ipynb                      # Code for Part 1. Sequence Tagging: NER, answers question 1.1
├── Q1_part2.ipynb                      # Code for Part 1. Sequence Tagging: NER, answers question 1.2
├── Q1_part3.ipynb                      # Code for Part 1. Sequence Tagging: NER, contains the model and answers question 1.3
├── Q2_model_average.ipynb              # Code for Part 2. Sentence-Level Categorization: Question Classification, contains the model that uses average pooling and answers question 2
├── Q2_model_last_word.ipynb            # Code for Part 2. Sentence-Level Categorization: Question Classification, contains the model that uses last word pooling and answers question 2
├── Q2_model_max_pooling.ipynb          # Code for Part 2. Sentence-Level Categorization: Question Classification, contains the model that uses max pooling and answers question 2
├── Q2_model_sum.ipynb                  # Code for Part 2. Sentence-Level Categorization: Question Classification, contains the model that uses sum pooling and answers question 2
├── Q2_preprocessing.ipynb              # Code for Part 2. Sentence-Level Categorization: Question Classification, contains preprocessing code
├── README.md                           # This file
├── requirements.txt                    # Contains the packages and libraries needed to run all the code
├── TREC_dataset                        # Contains TREC dataset from Kaggle
│   ├── modified_test_data.csv          # Modified TREC test dataset to combine 2 classes into 1
│   ├── modified_training_data.csv      # Modified TREC training dataset to combine 2 classes into 1
│   ├── test.csv                        # TREC test data. Replace this file if using own dataset, and rerun Q2_preprocessing.ipynb
│   └── train.csv                       # TREC training data. Replace this file if using own dataset, and rerun Q2_preprocessing.ipynb
└── CoNLL2003_dataset                   # Contains CoNLL2003 dataset
    ├── eng.testa                       # CoNLL2003 development (validation) dataset
    ├── eng.testb                       # CoNLL2003 test dataset
    └── eng.train                       # CoNLL2003 training dataset
```

The codes are given in the form of Jupyter notebooks. Files prepended with Q1_ contains code for Part 1 (Sequence Tagging: NER), while 
files prepended with Q2_ contains code for Part 2 (Sentence-Level Categorization: Question Classification).

Before running any of the Q2 models, one should run "Q2_preprocessing.ipynb" which will generate the updated dataset which will be used by the models. 
There are no restrictions on which files can be run first for Q1_ notebooks.

## Running the code

After

## Explanation of sample output

Part 1
Q1_part1.ipynb
Q1_part2.ipynb
Q1_part2.ipynb

Part 2
Q2_preprocessing.ipynb

Q2_model_average.ipynb
Q2_model_last_word.ipynb
Q2_model_max_pooling.ipynb
Q2_model_sum.ipynb

## Done by
Poh Yang Quan Eugene  
Wong Yi Pun  
Jeremy U Keat  
Ng Yue Jie Alphaeus  
Roy Lau Run-Xuan  

## Acknowledgements
CoNLL2023: https://github.com/TheAnig/NER-LSTM-CNN-Pytorch/tree/master/data  
TREC: https://www.kaggle.com/datasets/thedevastator/the-trec-question-classification-dataset-a-longi/
CS230 Stanford: https://github.com/cs230-stanford/cs230-code-examples/tree/master/pytorch/nlp
CS230 Stanford: https://cs230.stanford.edu/blog/namedentity/