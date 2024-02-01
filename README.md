# SC4002_NLP

## Setup instructions

The following instructions setups a new virtual environment for python and installs the needed libraries.  
These instructions assumes that the user is using a Windows machine. Ensure that Python 3.11 is installed as well, do not use Python 3.12. 
Open the command prompt in the root directory "SC4002_G3" and enter the following commands.

```cmd
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Alternatively, you can run the following instructions to setup a new virtual environment for python and installs the needed libraries.
This uses the default python version installed in the machine. However do ensure that the Python version is not 3.12.
```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Ensure that 2 directories are created in the "Source Code" folder, "CoNLL2003_dataset" and "TREC_dataset". For "CoNLL2003_dataset/", put "eng.testa", "eng.testb" and "eng.train" into it 
and for "TREC_dataset/" include the "test.csv" and "train.csv" from Kaggle. 

If the machine that is running the code has a GPU with cuda you can install torch with cuda support instead 
using the command given in the PyTorch documentation https://pytorch.org/get-started/locally/

## Folder structure and files 

The following is a tree structure that shows the files found in this folder SC4002_G3. A brief description of what each file is for is included at the side.

```markdown
SC4002_G3
├── SC4002_G3.pdf                         # Assignment report
├── README.txt                            # This file
├── requirements.txt                      # Contains the packages and libraries needed to run all the code
└── Source Code                           # Folder containing the source code
    ├── Q1_part3_f1_score_vs_epoch.png    # Plot of f1 score in each epoch for part 1
    ├── Q1_part1.ipynb                    # Code for Part 1. Sequence Tagging: NER, answers question 1.1
    ├── Q1_part2.ipynb                    # Code for Part 1. Sequence Tagging: NER, answers question 1.2
    ├── Q1_part3.ipynb                    # Code for Part 1. Sequence Tagging: NER, contains the model and answers question 1.3
    ├── Q2_model_average.ipynb            # Code for Part 2. Sentence-Level Categorization: Question Classification, contains the model that uses average pooling and answers question 2
    ├── Q2_model_last_word.ipynb          # Code for Part 2. Sentence-Level Categorization: Question Classification, contains the model that uses last word pooling and answers question 2
    ├── Q2_model_max_pooling.ipynb        # Code for Part 2. Sentence-Level Categorization: Question Classification, contains the model that uses max pooling and answers question 2
    ├── Q2_model_sum.ipynb                # Code for Part 2. Sentence-Level Categorization: Question Classification, contains the model that uses sum pooling and answers question 2
    ├── Q2_preprocessing.ipynb            # Code for Part 2. Sentence-Level Categorization: Question Classification, contains preprocessing code
    ├── TREC_dataset                      # Contains TREC dataset from Kaggle
    │   ├── modified_test_data.csv        # Modified TREC test dataset to combine 2 classes into 1
    │   ├── modified_training_data.csv    # Modified TREC training dataset to combine 2 classes into 1
    │   ├── test.csv                      # TREC test data. Replace this file if using own dataset, and rerun Q2_preprocessing.ipynb
    │   └── train.csv                     # TREC training data. Replace this file if using own dataset, and rerun Q2_preprocessing.ipynb
    └── CoNLL2003_dataset                 # Contains CoNLL2003 dataset
        ├── eng.testa                     # CoNLL2003 development (validation) dataset
        ├── eng.testb                     # CoNLL2003 test dataset
        └── eng.train                     # CoNLL2003 training dataset
```

The codes are given in the form of Jupyter notebooks. Files prepended with Q1_ contains code for Part 1 (Sequence Tagging: NER), while 
files prepended with Q2_ contains code for Part 2 (Sentence-Level Categorization: Question Classification).

Before running any of the Q2 models, one should run "Q2_preprocessing.ipynb" which will generate the updated dataset which will be used by the models. 
There are no restrictions on which files can be run first for Q1_ notebooks.

## Running the code

After starting up jupyter lab, a browser tab will open. In the left panel, double-click on the "Source Code" folder to enter it. Double-click on any 
notebook to open it in the right panel. You should be able to see the code now on the right panel. 

At the top left, click on "Run", then "Run All Cells" which will run all the code in that notebook.

## Explanation of sample output

### Part 1

#### Q1_part1.ipynb

The output of this file shows the most similar words to student, Apple and apple based on the word2vec embeddings. We also calculated the cosine distance 
for these 3 words for additional verification.

```markdown
(a) Most similar word to 'student': students, Cosine Similarity: 0.7294867038726807
(b) Most similar word to 'Apple': Apple_AAPL, Cosine Similarity: 0.7456986308097839
(c) Most similar word to 'apple': apples, Cosine Similarity: 0.720359742641449
```

#### Q1_part2.ipynb

The output of this file shows the size of the training (eng.train), development (eng.testa) and test file (eng.testb) for the CoNLL2003 dataset. 
All NER tags found in the datasets are printed as well.

```markdown
The number of sentences in the training data is 14041.
The number of sentences in the development data is 3250.
The number of sentences in the test data is 3453.
The set of NER tags are {'O', 'B-ORG', 'I-ORG', 'B-MISC', 'B-LOC', 'I-MISC', 'I-LOC', 'I-PER'}.
```

#### Q1_part3.ipynb

The neural network used is LSTM. 

We have also printed out the model parameters. 
```markdown
===================================================================
Layer                                         Parameters
===================================================================
lstm.weight_ih_l0 torch.Size([1200, 300])   : 1,200*300 = 360,000
lstm.weight_hh_l0 torch.Size([1200, 300])   : 1,200*300 = 360,000
lstm.bias_ih_l0 torch.Size([1200])          : 1,200
lstm.bias_hh_l0 torch.Size([1200])          : 1,200
batch_norm1.weight torch.Size([300])        : 300
batch_norm1.bias torch.Size([300])          : 300
fc.weight torch.Size([8, 300])              : 8*300 = 2,400
fc.bias torch.Size([8])                     : 8
===================================================================
Total parameters                            : 725,408
```

During each training epoch, we printed the f1 scores, and the loss for both the training and development set. This is shown in the report.

The image (Q1_part3_f1_score_vs_epoch.png) shows the plot of the f1 score (training and development) at each training epoch.

A sample question is chosen from the test set (eng.trainb) which we pass through the model to predict the NER tags.

```markdown
sample question: ['Yevgeny', 'Kafelnikov', 'UNK', 'Russia', 'UNK', 'beat', 'Jim', 'Courier', 'UNK', 'U.S.', 'UNK', 'UNK', 'UNK', 'UNK']
sample_label_predict: ['I-PER', 'I-PER', 'O', 'I-LOC', 'O', 'O', 'I-PER', 'I-PER', 'O', 'I-LOC', 'O', 'O', 'O', 'O']
sample_label_true: ['I-PER', 'I-PER', 'O', 'I-LOC', 'O', 'O', 'I-PER', 'I-PER', 'O', 'I-LOC', 'O', 'O', 'O', 'O']
```

Finally, we also printed out the classification report of the model on the test set, as well as the f1 score of the test set.

```markdown
Test set f1 score: 0.7427138285911962

model_class_report: 
              precision    recall  f1-score   support

         LOC       0.79      0.82      0.80      1661
        MISC       0.60      0.66      0.63       702
         ORG       0.62      0.70      0.66      1655
         PER       0.83      0.82      0.82      1611

   micro avg       0.72      0.76      0.74      5629
   macro avg       0.71      0.75      0.73      5629
weighted avg       0.73      0.76      0.74      5629
```

### Part 2
#### Q2_preprocessing.ipynb

The output of this file shows the classes that have been merged as well as the new mappings from the old classes to new classes. Additionally, we have also
printed the mapping from the new classes to the old classes.

#### Q2 models

For comparison, the models used in the four notebooks below does not change, only the aggregation method used. The model parameters are printed in the notebook and are
the following.

```markdown
===================================================================
Layer                                         Parameters
===================================================================
lstm.weight_ih_l0 torch.Size([1200, 300])   : 1,200*300 = 360,000
lstm.weight_hh_l0 torch.Size([1200, 300])   : 1,200*300 = 360,000
lstm.bias_ih_l0 torch.Size([1200])          : 1,200
lstm.bias_hh_l0 torch.Size([1200])          : 1,200
batch_norm1.weight torch.Size([300])        : 300
batch_norm1.bias torch.Size([300])          : 300
fc1.weight torch.Size([150, 300])           : 150*300 = 45,000
fc1.bias torch.Size([150])                  : 150
batch_norm2.weight torch.Size([150])        : 150
batch_norm2.bias torch.Size([150])          : 150
fc2.weight torch.Size([5, 150])             : 5*150 = 750
fc2.bias torch.Size([5])                    : 5
batch_norm3.weight torch.Size([5])          : 5
batch_norm3.bias torch.Size([5])            : 5
===================================================================
Total parameters                            : 769,215
```

#### Q2_model_average.ipynb

The neural network used is LSTM, and average pooling is used. 

During each training epoch, we printed the loss and accuracy on the development set.

Finally, we also printed out the accuracy on the test set, as well as predicted labels for some example sentences.

```markdown
final_test_accuracy=0.922

sentence = What is a squirrel?, label = 0
sentence = Is Singapore located in Southeast Asia?, label = 3
sentence = Is Singapore in China?, label = 1
sentence = Name 11 famous martyrs ., label = 4
sentence = What ISPs exist in the Caribbean ?, label = 4
sentence = How many cars are manufactured every day?, label = 4
```

#### Q2_model_last_word.ipynb

The neural network used is LSTM, and the last word representation is taken.

During each training epoch, we printed the loss and accuracy on the development set.

Finally, we also printed out the accuracy on the test set, as well as predicted labels for some example sentences.

```markdown
final_test_accuracy=0.9

sentence = What is a squirrel ?, label = 1
sentence = Is Singapore located in Southeast Asia ?, label = 0
sentence = Is Singapore in China ?, label = 0
sentence = Name 11 famous martyrs ., label = 0
sentence = What ISPs exist in the Caribbean ?, label = 1
sentence = How many cars are manufactured every day ?, label = 4
```

#### Q2_model_max_pooling.ipynb

The neural network used is LSTM, and max pooling is used.

During each training epoch, we printed the loss and accuracy on the development set.

Finally, we also printed out the accuracy on the test set, as well as predicted labels for some example sentences.

```markdown
final_test_accuracy=0.91

sentence = What is a squirrel ?, label = 0
sentence = Is Singapore located in Southeast Asia ?, label = 3
sentence = Is Singapore in China ?, label = 1
sentence = Name 11 famous martyrs ., label = 4
sentence = What ISPs exist in the Caribbean ?, label = 4
sentence = How many cars are manufactured every day ?, label = 4
```

#### Q2_model_sum.ipynb

The neural network used is LSTM, and sum pooling is used.

During each training epoch, we printed the loss and accuracy on the development set.

Finally, we also printed out the accuracy on the test set, as well as predicted labels for some example sentences.

```markdown
final_test_accuracy=0.92

sentence = What is a squirrel ?, label = 0
sentence = Is Singapore located in Southeast Asia ?, label = 1
sentence = Is Singapore in China ?, label = 1
sentence = Name 11 famous martyrs ., label = 4
sentence = What ISPs exist in the Caribbean ?, label = 4
sentence = How many cars are manufactured every day ?, label = 4
```

## Done by
[Poh Yang Quan Eugene](https://github.com/Eugene7997)
[Wong Yi Pun](https://github.com/ypwong99)
[Jeremy U Keat](https://github.com/jeremyu25)  
[Ng Yue Jie Alphaeus](https://github.com/AlphaeusNg)  
[Roy Lau Run-Xuan](https://github.com/roylau98)  

## Acknowledgements
CoNLL2023: https://github.com/TheAnig/NER-LSTM-CNN-Pytorch/tree/master/data

TREC: https://www.kaggle.com/datasets/thedevastator/the-trec-question-classification-dataset-a-longi/

CS230 Stanford: https://github.com/cs230-stanford/cs230-code-examples/tree/master/pytorch/nlp

CS230 Stanford: https://cs230.stanford.edu/blog/namedentity/
