# coding: utf-8
import fire
import numpy as np
from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F

from tea.config.app_cfg import AppConfig
import tea.models.factory as MFactory
from tea.data.text_loader import TextLoader
from tea.trainer.basic_learner import build_trainer
from tea.trainer.helper import explore_lr_and_plot
import matplotlib.pyplot as plt

import time
import math
import torch
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
            text, targets = batch
            output = model(text)
            output_flat = output.view(-1, ntokens)
            total_loss += loss_fn(output_flat, targets.view(-1)).item()
    return total_loss / len(data_source)


def train(cfg, model, train_ds, epoch, lr, ntokens, loss_fn, clip, log_freq=200):
    # Turn on training mode which enables dropout.
    batch_sz = cfg.get_batch_sz()

    model.train()
    total_loss = 0.
    start_time = time.time()
    for i, batch in enumerate(train_ds):
        text, targets = batch

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
                epoch, i, len(train_ds), lr,
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
    train_iter, valid_iter = data.BPTTIterator.splits(
        (train, valid), batch_size=cfg.get_batch_sz(), bptt_len=cfg.get_bptt(), device=device)

    test_iter, = data.BPTTIterator.splits(
        (test,), batch_size=1, bptt_len=cfg.get_bptt(), device=device)

    print('Train/valid/test', len(train_iter), len(valid_iter), len(test_iter))
    return TextLoader(train_iter), TextLoader(valid_iter), TextLoader(test_iter)


def word_ids_to_sentence(id_tensor, vocab, join=None):
    """Converts a sequence of word ids to a sentence"""
    if isinstance(id_tensor, torch.Tensor):
        ids = id_tensor.transpose(0, 1).contiguous().view(-1)
    elif isinstance(id_tensor, np.ndarray):
        ids = id_tensor.transpose().reshape(-1)
    else:
        raise Exception(f"Unexcepted type {type(id_tensor)}")
    batch = [vocab.itos[ind] for ind in ids] # denumericalize
    if join is None:
        return batch
    else:
        return join.join(batch)


def loss_fn(y_pred, y):
    sizes = y_pred.size()
    loss = F.cross_entropy(y_pred.view(-1, sizes[-1]), y.view(-1))
    return loss


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
        epochs=30,
        emsize=200,
#        lr=20.0,
        lr=1e-3,
        clip=0.25,
        batch_sz=40,
        bppt=70,
        log_freq=200,
        use_gpu=True,
        explore_lr=False):
    # Step 1: parse config
    cfg = AppConfig.from_file(ini_file,
                                model_cfg=model_cfg,
                                model_out_dir=model_out_dir,
                                epochs=epochs,
                                emsize=emsize,
                                lr=lr,
                                clip=clip,
                                batch_sz=batch_sz,
                                log_freq=log_freq,
                                bppt=bppt,
                                use_gpu=use_gpu)
    cfg.print()
    # Step 2: create data sets and loaders
    TEXT = data.Field(lower=True, batch_first=True)
    train_ds, valid_ds, test_ds= build_train_val_test_datasets(cfg, TEXT, emsize)
    vocab_sz = len(TEXT.vocab)
    print('len(TEXT.vocab)', vocab_sz)

    # Step 3: create model

    # add extra context
    cfg.update(vocab_sz= vocab_sz)
    model = MFactory.create_model(cfg, init_params=False)
    print(model)
    model.tie_weights()
    model.init_params(initrange=0.01)

    # Step 4: train/valid
    learner = build_trainer(cfg, model)

    # Step 5: optionally find the best lr
    if explore_lr:
        path = learner.cfg.get_model_out_dir()
        path = Path(path) / 'lr_tmp.pch'
        lr = explore_lr_and_plot(learner, train_ds, path, loss_fn=loss_fn, start_lr=1.0e-4, end_lr=100.0, batches=100)
        print(f'Idea lr {lr}')
        plt.show()
    else:
        # accuracy is a classification metric
        # metrics = {"accuracy": Accuracy()}
        learner.fit(train_ds, valid_ds, loss_fn=loss_fn, metrics={})

    # # model = create_model(TEXT, model_str, vocab_sz, emsize, nhid, nlayers, dropout, tied)
    # device = cfg.get_device()
    # model = model.to(device)
    #
    # loss_fn = nn.CrossEntropyLoss()
    #
    # # Loop over epochs.
    # lr = cfg.get_lr()
    # best_val_loss = None
    # epochs = cfg.get_epochs()
    # save_path = 'model.pt'
    #
    # # At any point you can hit Ctrl + C to break out of training early.
    # try:
    #     for epoch in range(1, epochs+1):
    #         epoch_start_time = time.time()
    #         train(cfg, model, train_ds, epoch, lr, vocab_sz, loss_fn, clip, log_freq)
    #         val_loss = evaluate(cfg, model, valid_ds, vocab_sz, loss_fn)
    #         print('-' * 89)
    #         print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
    #                 'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
    #                                            val_loss, math.exp(val_loss)))
    #         print('-' * 89)
    #         # Save the model if the validation loss is the best we've seen so far.
    #         if not best_val_loss or val_loss < best_val_loss:
    #             with open(save_path, 'wb') as f:
    #                 torch.save(model, f)
    #             best_val_loss = val_loss
    #         else:
    #             # Anneal the learning rate if no improvement has been seen in the validation dataset.
    #             lr /= 4.0
    # except KeyboardInterrupt:
    #     print('-' * 89)
    #     print('Exiting from training early')
    #
    # # Load the best saved model.
    # with open(save_path, 'rb') as f:
    #     model = torch.load(f)
    #
    # # Run on test data.
    # test_loss = evaluate(cfg, model, test_ds, vocab_sz, loss_fn)
    # print('=' * 89)
    # print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    #     test_loss, math.exp(test_loss)))
    # print('=' * 89)
    #
    # sentence = next(iter(test_ds)); vars(sentence).keys()
    # print(f"Test sentence: {word_ids_to_sentence(sentence.text, TEXT.vocab, join=' ')}")
    # pred = model(sentence.text).cpu().data.numpy()
    # pred_s = word_ids_to_sentence(np.argmax(pred, axis=2), TEXT.vocab, join=' ')
    # print(f'Generated sentence: {pred_s}')


if __name__ == '__main__':
    fire.Fire(run)
