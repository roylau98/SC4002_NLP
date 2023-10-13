# SC4002_NLP

### convert this to a READMD.txt later  

## Setup instructions

There are two different ways of installing dependencies needed for this project.  

### Using venv
The following instructions setups a new virtual environment for python and installs the needed libraries.  
These instructions assumes that the user is using a Windows machine.  

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Using poetry
The following instructions setups a new poetry environment and installs the needed libraries. 
Ensure that poetry is installed in your machine (https://python-poetry.org/docs/#installation), and you are using a Python version >= 3.9.  

```cmd
poetry shell
poetry install 
poetry run main.py
```

## Done by
Poh Yang Quan Eugene  
Wong Yi Pun  
Jeremy U Keat  
Ng Yue Jie Alphaeus  
Roy Lau Run-Xuan  

## Acknowledgements
CoNLL2023: https://github.com/TheAnig/NER-LSTM-CNN-Pytorch/tree/master/data  
TREC: https://www.kaggle.com/datasets/thedevastator/the-trec-question-classification-dataset-a-longi/  