import torch
import torch.nn as nn
from torchtext import data, datasets
from torchtext.vocab import GloVe
import torch.optim as optim
from torch.autograd import Variable as V
from torchtext.datasets import WikiText2
import torch.nn.functional as F
import numpy as np


class RNNModel(nn.Module):
    def __init__(self, ntoken, ninp,
                 nhid, nlayers, bsz,
                 dropout=0.5, tie_weights=True):
        super(RNNModel, self).__init__()
        self.nhid, self.nlayers, self.bsz = nhid, nlayers, bsz
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        self.init_weights()
        self.hidden = self.init_hidden(bsz)  # the input is a batched consecutive corpus
        # therefore, we retain the hidden state across batches

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input):
        emb = self.drop(self.encoder(input))
        output, self.hidden = self.rnn(emb, self.hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1))

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (V(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                V(weight.new(self.nlayers, bsz, self.nhid).zero_()))

    def reset_history(self):
        self.hidden = tuple(V(v.data) for v in self.hidden)

# Approach 1:
# set up fields
TEXT = data.Field(lower=True, batch_first=True)

# make splits for data
train, valid, test = datasets.WikiText2.splits(TEXT)

# print information about the data
print('train.fields', train.fields)
print('len(train)', len(train))
print('vars(train[0])', vars(train[0])['text'][0:10])

emsize = 200

# build the vocabulary
TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=emsize))

# print vocab information
print('len(TEXT.vocab)', len(TEXT.vocab))

ntokens = len(TEXT.vocab)
nhid = 200
nlayers = 2
dropout = 0.2
tied = 2
device = "cpu"
batch_size = 20
bptt = 35

# make iterator for splits
train_iter, valid_iter, test_iter = data.BPTTIterator.splits(
    (train, valid, test), batch_size=batch_size, bptt_len=bptt, device=device)

# # print batch information
# batch = next(iter(train_iter))
# print(batch.text)
# print(batch.target)

weight_matrix = TEXT.vocab.vectors
model = RNNModel(weight_matrix.size(0),
                 weight_matrix.size(1), nhid, nlayers, batch_size)

model.encoder.weight.data.copy_(weight_matrix)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.7, 0.99))
n_tokens = weight_matrix.size(0)


def train_epoch(epoch):
    """One epoch of a training loop"""
    epoch_loss = 0
    for batch in train_iter:
        # reset the hidden state or else the model will try to backpropagate to the
        # beginning of the dataset, requiring lots of time and a lot of memory
        model.reset_history()

    optimizer.zero_grad()

    text, targets = batch.text, batch.target
    prediction = model(text)
    # pytorch currently only supports cross entropy loss for inputs of 2 or 4 dimensions.
    # we therefore flatten the predictions out across the batch axis so that it becomes
    # shape (batch_size * sequence_length, n_tokens)
    # in accordance to this, we reshape the targets to be
    # shape (batch_size * sequence_length)
    loss = criterion(prediction.view(-1, n_tokens), targets.view(-1))
    loss.backward()

    optimizer.step()

    epoch_loss += loss.item() * prediction.size(0) * prediction.size(1)

    epoch_loss /= len(train.examples[0].text)

    # monitor the loss
    val_loss = 0
    model.eval()
    for batch in valid_iter:
        model.reset_history()
        text, targets = batch.text, batch.target
        prediction = model(text)
        loss = criterion(prediction.view(-1, n_tokens), targets.view(-1))
        val_loss += loss.item() * text.size(0)
    val_loss /= len(valid.examples[0].text)

    print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch, epoch_loss, val_loss))

n_epochs = 2
for epoch in range(1, n_epochs + 1):
    train_epoch(epoch)

def word_ids_to_sentence(id_tensor, vocab, join=None):
    """Converts a sequence of word ids to a sentence"""
    if isinstance(id_tensor, torch.LongTensor):
        ids = id_tensor.transpose(0, 1).contiguous().view(-1)
    elif isinstance(id_tensor, np.ndarray):
        ids = id_tensor.transpose().reshape(-1)
    batch = [vocab.itos[ind] for ind in ids] # denumericalize
    if join is None:
        return batch
    else:
        return join.join(batch)

b = next(iter(train_iter)); vars(b).keys()
arrs = model(b.text).cpu().data.numpy()
word_ids_to_sentence(np.argmax(arrs, axis=2), TEXT.vocab, join=' ')