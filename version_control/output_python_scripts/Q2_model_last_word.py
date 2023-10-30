#!/usr/bin/env python
# coding: utf-8

# # Import libraries

# In[1]:


import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange
from torch.autograd import Variable
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from nltk.tokenize import word_tokenize
import gensim.downloader

from seqeval.metrics import f1_score as f1_score_seqeval
from seqeval.metrics import classification_report
from seqeval.scheme import IOB1


# In[2]:


word2vec_goog1e_news: gensim.models.keyedvectors.KeyedVectors = gensim.downloader.load('word2vec-google-news-300')
word2vec_goog1e_news.add_vector("<pad>", np.zeros(300))
pad_index = word2vec_goog1e_news.key_to_index["<pad>"]
embedding_weights = torch.FloatTensor(word2vec_goog1e_news.vectors)
vocab = word2vec_goog1e_news.key_to_index


# # Import Dataset

# In[3]:


def tokenize_pd_series_to_lsit(list_of_text):
    tokenized = []
    for sentence in list_of_text:
        tokenized.append(word_tokenize(sentence.lower()))
    return tokenized

def format_label(label):
    return torch.unsqueeze(torch.tensor(label.to_list()), axis=1).tolist()

def indexify(data):
    setences = []
    for sentence in data:
        s = [vocab[token] if token in vocab
            else vocab['UNK']
            for token in sentence]
        setences.append(s)
    return setences


# In[4]:


training_data = pd.read_csv(filepath_or_buffer="TREC_dataset/modified_training_data.csv", sep=",") 
test_data = pd.read_csv(filepath_or_buffer="TREC_dataset/modified_test_data.csv", sep=",")

X = training_data["text"]
y = training_data["label-coarse"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=500)

X_test = test_data["text"]
y_test = test_data["label-coarse"]

X_train_lst = X_train.to_list()
X_val_lst = X_val.to_list()
X_test_lst = X_test.to_list()

X_train_tokenized = tokenize_pd_series_to_lsit(X_train_lst)
X_val_tokenized = tokenize_pd_series_to_lsit(X_val_lst)
X_test_tokenized = tokenize_pd_series_to_lsit(X_test_lst)

no_of_labels = max(y_train.to_list()) + 1


# In[5]:


X_train_tokenized_indexified = indexify(X_train_tokenized)
X_val_tokenized_indexified = indexify(X_val_tokenized)
X_test_tokenized_indexified = indexify(X_test_tokenized)

y_train_formatted = format_label(y_train)
y_val_formatted = format_label(y_val)
y_test_formatted = format_label(y_test)


# In[6]:


def data_iterator(sentences, labels, total_size: int, batch_size: int, shuffle: bool=False):
    # make a list that decides the order in which we go over the data- this avoids explicit shuffling of data
    order = list(range(total_size))
    if shuffle:
        random.seed(230)
        random.shuffle(order)

    # one pass over data
    for i in range((total_size+1)//batch_size):
        # fetch sentences and tags
        batch_sentences = [sentences[idx] for idx in order[i*batch_size:(i+1)*batch_size]]
        batch_tags = [labels[idx] for idx in order[i*batch_size:(i+1)*batch_size]]

        # compute length of longest sentence in batch
        batch_max_len = max([len(s) for s in batch_sentences])

        # prepare a numpy array with the data, initialising the data with pad_ind and all labels with -1
        # initialising labels to -1 differentiates tokens with tags from PADding tokens
        batch_data = vocab['<pad>']*np.ones((len(batch_sentences), batch_max_len))
        # batch_labels = -1*np.ones((len(batch_sentences), batch_max_len))
        batch_labels = -1*np.ones(len(batch_sentences))
        batch_labels = batch_labels.reshape(-1, 1)

        # print(f"batch_data.shape = {batch_data.shape}")
        # print(f"batch_labels.shape = {batch_labels.shape}")

        # copy the data to the numpy array
        for j in range(len(batch_sentences)):
            cur_len = len(batch_sentences[j])
            batch_data[j][:cur_len] = batch_sentences[j]
            batch_labels[j][:cur_len] = batch_tags[j]

        # since all data are indices, we convert them to torch LongTensors
        batch_data, batch_labels = torch.LongTensor(batch_data), torch.LongTensor(batch_labels)
        
        # convert them to Variables to record operations in the computational graph
        batch_data, batch_labels = Variable(batch_data), Variable(batch_labels)

        yield batch_data, batch_labels, batch_sentences


# In[7]:


class Net(nn.Module):
    """
    This is the standard way to define your own network in PyTorch. You typically choose the components
    (e.g. LSTMs, linear layers etc.) of your network in the __init__ function. You then apply these layers
    on the input step-by-step in the forward function. You can use torch.nn.functional to apply functions
    such as F.relu, F.sigmoid, F.softmax. Be careful to ensure your dimensions are correct after each step.

    You are encouraged to have a look at the network in pytorch/vision/model/net.py to get a better sense of how
    you can go about defining your own network.

    The documentation for all the various components available to you is here: http://pytorch.org/docs/master/nn.html
    """

    def __init__(self, embedding_weights, embedding_dim, lstm_hidden_dim, number_of_tags):
        """
        We define an recurrent network that predicts the NER tags for each token in the sentence. The components
        required are:

        - an embedding layer: this layer maps each index in range(params.vocab_size) to a params.embedding_dim vector
        - lstm: applying the LSTM on the sequential input returns an output for each token in the sentence
        - fc: a fully connected layer that converts the LSTM output for each token to a distribution over NER tags

        Args:
            params: (Params) contains vocab_size, embedding_dim, lstm_hidden_dim
        """
        super(Net, self).__init__()

        # the embedding takes as input the vocab_size and the embedding_dim
        # self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding = nn.Embedding.from_pretrained(embedding_weights, padding_idx=pad_index)

        # the LSTM takes as input the size of its input (embedding_dim), its hidden size
        # for more details on how to use it, check out the documentation
        self.lstm = nn.LSTM(embedding_dim,
                            lstm_hidden_dim, batch_first=True)

        # the fully connected layer transforms the output to give the final output layer
        self.fc = nn.Linear(lstm_hidden_dim, number_of_tags)

    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: (Variable) contains a batch of sentences, of dimension batch_size x seq_len, where seq_len is
               the length of the longest sentence in the batch. For sentences shorter than seq_len, the remaining
               tokens are PADding tokens. Each row is a sentence with each element corresponding to the index of
               the token in the vocab.

        Returns:
            out: (Variable) dimension batch_size*seq_len x num_tags with the log probabilities of tokens for each token
                 of each sentence.

        Note: the dimensions after each step are provided
        """
        #                                -> batch_size x seq_len
        # apply the embedding layer that maps each token to its embedding
        # dim: batch_size x seq_len x embedding_dim
        s = self.embedding(s)

        # run the LSTM along the sentences of length seq_len
        # dim: batch_size x seq_len x lstm_hidden_dim
        s, _ = self.lstm(s)
        # make the Variable contiguous in memory (a PyTorch artefact)
        # s = s.contiguous()
        # reshape the Variable so that each row contains one token
        # dim: batch_size*seq_len x lstm_hidden_dim
        # s = s.view(-1, s.shape[2])

        # Changed
        s = s[:, -1, :]
        
        # apply the fully connected layer and obtain the output (before softmax) for each token
        s = self.fc(s)                   # dim: batch_size*seq_len x num_tags

        # apply log softmax on each token's output (this is recommended over applying softmax
        # since it is numerically more stable)
        # return F.log_softmax(s, dim=1)   # dim: batch_size*seq_len x num_tags
        return F.softmax(s, dim=1)   # dim: batch_size*seq_len x num_tags


# In[8]:


def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs from the model and labels for all tokens. Exclude loss terms
    for PADding tokens.

    Args:
        outputs: (Variable) dimension batch_size*seq_len x num_tags - log softmax output of the model
        labels: (Variable) dimension batch_size x seq_len where each element is either a label in [0, 1, ... num_tag-1],
                or -1 in case it is a PADding token.

    Returns:
        loss: (Variable) cross entropy loss for all tokens in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """

    # reshape labels to give a flat vector of length batch_size*seq_len
    labels = labels.view(-1)

    # since PADding tokens have label -1, we can generate a mask to exclude the loss from those terms
    mask = (labels >= 0).float()

    # indexing with negative values is not supported. Since PADded tokens have label -1, we convert them to a positive
    # number. This does not affect training, since we ignore the PADded tokens with the mask.
    labels = labels % outputs.shape[1]
    num_tokens = int(torch.sum(mask))

    # compute cross entropy loss for all tokens (except PADding tokens), by multiplying with mask.
    
    # return -torch.sum(outputs[range(outputs.shape[0]), labels]*mask)/ torch.sum(num_tokens)
    a = outputs[range(outputs.shape[0]), labels]*mask
    b = -torch.sum(a)
    c = num_tokens
    return b/c


# In[9]:


class RunningAverage:
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


# In[10]:


def train(model, optimizer, loss_fn, data_iterator, metrics, num_steps):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        data_iterator: (generator) a generator that generates batches of data and labels
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = RunningAverage()

    # Use tqdm for progress bar
    t = trange(num_steps)
    for i in t:
        # fetch the next training batch
        train_batch, labels_batch, _ = next(data_iterator)

        # compute model output and loss
        output_batch = model(train_batch)
        loss = loss_fn(output_batch, labels_batch)

        # clear previous gradients, compute gradients of all variables wrt loss
        optimizer.zero_grad()
        loss.backward()

        # performs updates using calculated gradients
        optimizer.step()

        # Evaluate summaries only once in a while
        if i % 10 == 0:
            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()

            # compute all metrics on this batch
            # summary_batch = {metric: metrics[metric](output_batch, labels_batch)
            #                  for metric in metrics}
            # summary_batch['loss'] = loss.item()
            # summ.append(summary_batch)

        # update the average loss
        loss_avg.update(loss.item())
        t.set_postfix(loss='{:05.3f}'.format(loss_avg()))

    # compute mean of all metrics in summary
    # metrics_mean = {metric: np.mean([x[metric]
    #                                  for x in summ]) for metric in summ[0]}
    # metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
    #                             for k, v in metrics_mean.items())
    # print("- Train metrics: " + metrics_string)


# In[11]:


def evaluate(model, loss_fn, data_iterator, metrics, num_steps):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        data_iterator: (generator) a generator that generates batches of data and labels
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []

    # compute metrics over the dataset
    for _ in range(num_steps):
        # fetch the next evaluation batch
        data_batch, labels_batch, _ = next(data_iterator)

        # compute model output
        output_batch = model(data_batch)
        loss = loss_fn(output_batch, labels_batch)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()

        # compute all metrics on this batch
        # summary_batch = {metric: metrics[metric](output_batch, labels_batch)
        #                  for metric in metrics}
        # summary_batch['loss'] = loss.item()
        # summ.append(summary_batch)

    # compute mean of all metrics in summary
    # print(f"summ: {summ}")
    # metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    # metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    # print("- Eval metrics : " + metrics_string)
    # return metrics_mean


# In[12]:


def train_and_evaluate(
        model,
        train_sentences,
        train_labels,
        val_sentences,
        val_labels,
        num_epochs: int,
        batch_size: int,
        optimizer,
        loss_fn,
        metrics
):
    for epoch in range(num_epochs):
        # Run one epoch
        print("Epoch {}/{}".format(epoch + 1, num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        num_steps = (len(train_sentences) + 1) // batch_size
        train_data_iterator = data_iterator(
            train_sentences, train_labels, len(train_sentences), batch_size, shuffle=True)
        train(model, optimizer, loss_fn, train_data_iterator,
              metrics, num_steps)

        # Evaluate for one epoch on validation set
        num_steps = (len(val_sentences) + 1) // batch_size
        val_data_iterator = data_iterator(
            val_sentences, val_labels, len(val_sentences), batch_size, shuffle=False)
        val_metrics = evaluate(
            model, loss_fn, val_data_iterator, metrics, num_steps)
   


# In[13]:


inv_vocab = {v: k for k, v in vocab.items()}

def id_to_words(sentence):
    new_sentence = [inv_vocab[i] for i in sentence]
    return new_sentence


# In[14]:


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all tokens. Exclude PADding terms.

    Args:
        outputs: (np.ndarray) dimension batch_size*seq_len x num_tags - log softmax output of the model
        labels: (np.ndarray) dimension batch_size x seq_len where each element is either a label in
                [0, 1, ... num_tag-1], or -1 in case it is a PADding token.

    Returns: (float) accuracy in [0,1]
    """

    # reshape labels to give a flat vector of length batch_size*seq_len
    labels = labels.ravel()

    # since PADding tokens have label -1, we can generate a mask to exclude the loss from those terms
    mask = (labels >= 0)

    # np.argmax gives us the class predicted for each token by the model
    outputs = np.argmax(outputs, axis=1)


    # compare outputs with labels and divide by number of tokens (excluding PADding tokens)
    return np.sum(outputs == labels)/float(np.sum(mask))



def calculate_multiclass_f1_score(outputs, labels):
    
    labels = labels.ravel()
    mask = (labels >= 0)  
    outputs = np.argmax(outputs, axis=1)
    outputs = outputs[mask]
    labels = labels[mask]
    outputs = outputs
    labels = labels
    outputs = np.expand_dims(outputs, axis=0)
    labels = np.expand_dims(labels, axis=0)
    outputs = outputs.tolist()
    labels = labels.tolist()
    f1= f1_score_seqeval(labels, outputs, mode='strict', scheme=IOB1)
    return f1

def classification_report_gen(outputs, labels):
    labels = labels.ravel()
    mask = (labels >= 0)  
    outputs = np.argmax(outputs, axis=1)
    outputs = outputs[mask]
    labels = labels[mask]
    outputs = outputs
    labels = labels
    outputs = np.expand_dims(outputs, axis=0)
    labels = np.expand_dims(labels, axis=0)
    outputs = outputs.tolist()
    labels = labels.tolist()
    return classification_report(labels, outputs, mode='strict', scheme=IOB1)
    
def calculate_multiclass_f1_score2(outputs, labels):
    
    labels = labels.ravel()
    mask = (labels >= 0)  
    outputs = np.argmax(outputs, axis=1)
    outputs = outputs[mask]
    labels = labels[mask]
    f1= f1_score(labels, outputs, average='macro')
    return f1

def calculate_multiclass_f1_score3(outputs, labels):
    
    labels = labels.ravel()
    mask = (labels >= 0)  
    outputs = np.argmax(outputs, axis=1)
    outputs = outputs[mask]
    labels = labels[mask]
    f1= f1_score(labels, outputs, average='micro')
    return f1


metrics = {
    'f1_seqeval': calculate_multiclass_f1_score,
    'f1_micro': calculate_multiclass_f1_score3,
    'f1 macro': calculate_multiclass_f1_score2,
    'accuracy': accuracy
    # could add more metrics such as accuracy for each token type
}


# In[15]:


import warnings
warnings.filterwarnings('ignore')


# In[22]:


# manually change vocab size (unique no. of words) and change label size (unique no. of labels) for now
model = Net(embedding_weights, 300, 300, no_of_labels)
optimizer = optim.Adam(model.parameters(), lr=0.01)

if (os.path.isfile("model_weights2.pth")):
    model.load_state_dict(torch.load('model_weights2.pth'))
else:
    train_and_evaluate(model, X_train_tokenized_indexified , y_train_formatted , X_val_tokenized_indexified  , y_val_formatted , 10, 5, optimizer, loss_fn, metrics)
    torch.save(model.state_dict(), 'model_weights2.pth')


# In[23]:


'''Test batch- tensor of n_sentences x max_len_sentence
   Labels_batch- tensor of n_sentences x max_len_sentence
   Test sentences- list of n_sentences x sentence_length(no padding)'''
print(len(X_test_tokenized_indexified))
test_data_iterator = data_iterator(X_test_tokenized_indexified , y_test_formatted , len(X_test_tokenized_indexified), len(X_test_tokenized_indexified), shuffle=True)
test_batch, labels_batch, test_sentences = next(test_data_iterator)


# In[24]:


print(f"type test: {type(test_batch)}")
print(f"len test: {len(test_batch)}")
print(f"type label: {type(labels_batch)}")
print(f"len label: {len(labels_batch)}")


# In[25]:


model_output = model(test_batch)


# In[26]:


model_output_numpy = model_output.detach().numpy()
labels_batch_numpy = labels_batch.detach().numpy()


# In[27]:


f1_score_seqeval = calculate_multiclass_f1_score(model_output_numpy, labels_batch_numpy)
f1_score_macro = calculate_multiclass_f1_score2(model_output_numpy, labels_batch_numpy)
f1_score_micro = calculate_multiclass_f1_score3(model_output_numpy, labels_batch_numpy)


# In[ ]:


print(f"f1_score_seqeval: {f1_score_seqeval}")
print(f"f1_score_macro: {f1_score_macro}")
print(f"f1_score_micro: {f1_score_micro}")


# In[ ]:


model_class_report = classification_report_gen(model_output_numpy, labels_batch_numpy)
print(f"model_class_report: \n{model_class_report}")


# In[ ]:





# In[ ]:


predicted_labels = np.argmax(model_output.detach().numpy(), axis=1)
print(f"sentences_w_words \n {len(test_sentences)}")
print(f"predicted_labels \n {len(predicted_labels)}")


# In[ ]:


sample_output = model(test_batch[10].unsqueeze(0))
id_to_words(test_sentences[10])


# In[ ]:


sample_mask = (labels_batch[10] >= 0)
sample_label_predict = np.argmax(sample_output.detach().numpy(), axis=1)[sample_mask]
sample_label_true = labels_batch[10][sample_mask]


# In[ ]:


print(f"sample_label_predict: {sample_label_predict}")
print(f"sample_label_true: {sample_label_true.numpy()}")


# In[ ]:




