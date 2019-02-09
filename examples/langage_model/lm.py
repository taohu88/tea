# coding: utf-8
import fire
import numpy as np
from pathlib import Path

from tea.utils.commons import detach_all
from tea.metrics.accuracy import Accuracy
from tea.vision.cv import transforms
from tea.config.app_cfg import AppConfig
import tea.data.data_loader_factory as DLFactory
import tea.models.factory as MFactory
from tea.trainer.base_learner import build_trainer
from tea.trainer.helper import explore_lr_and_plot
# import matplotlib.pyplot as plt

import time
import math
import torch
import torch.nn as nn
import torch.onnx
from torchtext import data, datasets
from torchtext.vocab import GloVe


###############################################################################
# Training code
###############################################################################
def evaluate(cfg, model, data_source, ntokens, loss_fn):
    # Turn on evaluation mode which disables dropout.
    batch_sz = cfg.get_batch_sz()

    model.eval()
    total_loss = 0.
    model.reset_context()
    with torch.no_grad():
        for batch in data_source:
            text, targets = batch.text, batch.target
            output = model(text)
            output_flat = output.view(-1, ntokens)
            total_loss += loss_fn(output_flat, targets.view(-1)).item()
    return total_loss / len(data_source)


def train(cfg, model, train_iter, epoch, lr, ntokens, loss_fn, clip, log_freq=200):
    # Turn on training mode which enables dropout.
    batch_sz = cfg.get_batch_sz()

    model.train()
    total_loss = 0.
    start_time = time.time()
    for i, batch in enumerate(train_iter):
        text, targets = batch.text, batch.target

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.reset_context()
        model.zero_grad()
        output = model(text)
        loss = loss_fn(output.view(-1, ntokens), targets.view(-1))
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()

        if i % log_freq == 0 and i > 0:
            cur_loss = total_loss / log_freq
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, i, len(train_iter), lr,
                elapsed * 1000 / log_freq, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def export_onnx(path, model, device, batch_size, seq_len):
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    torch.onnx.export(model, dummy_input, path)


def build_train_val_test_datasets(cfg, TEXT, emsize):
    device = cfg.get_device()

    # make splits for data
    train, valid, test = datasets.WikiText2.splits(TEXT)
    TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=emsize))
    train_iter, valid_iter, test_iter = data.BPTTIterator.splits(
        (train, valid, test), batch_size=cfg.get_batch_sz(), bptt_len=cfg.get_bptt(), device=device)

    print('Train/valid/test', len(train_iter), len(valid_iter), len(test_iter))
    return train_iter, valid_iter, test_iter


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


"""
It is good to follow pattern.
In this case, any application starts with cfg file, 
with optional override arguments like the following: 
    model_cfg
    model_out_dir
    epochs, lr, batch etc
"""
def run(ini_file='lm.ini',
        model_cfg='../cfg/lm-simple.cfg',
        model_out_dir='./models',
        epochs=1,
        lr=20.0,
        batch_sz=40,
        bppt=70,
        log_freq=200,
        use_gpu=True):
    # Step 1: parse config
    cfg = AppConfig.from_file(ini_file,
                        model_cfg=model_cfg,
                        model_out_dir=model_out_dir,
                        epochs=epochs,
                        lr=lr,
                        batch_sz=batch_sz,
                        log_freq=log_freq,
                        bppt=bppt,
                        use_gpu=use_gpu)
    cfg.print()

    emsize = 200
    clip = cfg.get_clip()

    # Step 2: create data sets and loaders
    TEXT = data.Field(lower=True, batch_first=True)
    train_iter, valid_iter, test_iter= build_train_val_test_datasets(cfg, TEXT, emsize)
    vocab_sz = len(TEXT.vocab)
    print('len(TEXT.vocab)', vocab_sz)

    # Step 3: create model

    # add extra context
    cfg.update(vocab_sz= vocab_sz)
    model = MFactory.create_model(cfg, init_params=False)
    print(model)
    model.tie_weights()
    model.init_params(initrange=0.01)


    # model = create_model(TEXT, model_str, vocab_sz, emsize, nhid, nlayers, dropout, tied)
    device = cfg.get_device()
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    # Loop over epochs.
    lr = cfg.get_lr()
    best_val_loss = None
    epochs = cfg.get_epochs()
    save_path = 'model.pt'

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, epochs+1):
            epoch_start_time = time.time()
            train(cfg, model, train_iter, epoch, lr, vocab_sz, loss_fn, clip, log_freq)
            val_loss = evaluate(cfg, model, valid_iter, vocab_sz, loss_fn)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss, math.exp(val_loss)))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(save_path, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 4.0
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(save_path, 'rb') as f:
        model = torch.load(f)

    # Run on test data.
    test_loss = evaluate(cfg, model, test_iter, vocab_sz, loss_fn)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)

    sentence = next(iter(test_iter)); vars(sentence).keys()
    print('Test sentence', sentence)
    pred = model(sentence.text).cpu().data.numpy()
    pred_s = word_ids_to_sentence(np.argmax(pred, axis=2), TEXT.vocab, join=' ')
    print('Predictions: ', pred_s)

if __name__ == '__main__':
    fire.Fire(run)
