# Import libraries.
from prototypical_loss import prototypical_loss as loss_fn
from parser_util import get_parser

import ipdb

from tqdm import tqdm
import numpy as np
import torch
import os
from utils import (init_seed, init_embedding_dataloader, init_embeddingnet,
                   init_optim, init_lr_scheduler, save_list_to_file)
                   #init_dataloader, init_protonet, 
                   
from setproctitle import setproctitle

def train(opt, trn_dataloader, model, optim, lr_scheduler, val_dataloader=None):
    '''
    Train the model with the prototypical learning algorithm
    '''
    # ipdb.set_trace()
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'

    if val_dataloader is None:
        best_state = None
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0

    for epoch in range(opt.epochs):
        print('=== Epoch: {} ==='.format(epoch))
        tr_iter = iter(trn_dataloader)
        model.train()
        for batch in tqdm(tr_iter):
            optim.zero_grad()
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            loss, acc = loss_fn(model_output, target=y,
                                n_support=opt.num_support_tr)
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
            train_acc.append(acc.item())
        avg_loss = np.mean(train_loss[-opt.iterations:])
        avg_acc = np.mean(train_acc[-opt.iterations:])
        print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))
        lr_scheduler.step()

        if val_dataloader is None:
            continue
        
        val_iter = iter(val_dataloader)
        model.eval()
        for batch in val_iter:
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            loss, acc = loss_fn(model_output, target=y,
                                n_support=opt.num_support_val)
            val_loss.append(loss.item())
            val_acc.append(acc.item())
        avg_loss = np.mean(val_loss[-opt.iterations:])
        avg_acc = np.mean(val_acc[-opt.iterations:])
        postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(
            best_acc)
        print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(
            avg_loss, avg_acc, postfix))
        if avg_acc >= best_acc:
            best_acc = avg_acc
            best_state = model.state_dict()

    for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
        save_list_to_file(os.path.join(opt.experiment_root,
                                       name + '.txt'), locals()[name])

    return best_state, best_acc, train_loss, train_acc, val_loss, val_acc


def main():
    '''
    Initialize everything and train
    '''
    options = get_parser().parse_args()
    setproctitle(options.exp_name)
    if not os.path.exists(options.experiment_root):
        os.makedirs(options.experiment_root)
    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)

    #ipdb.set_trace()
    embedding_trn_dataloader = init_embedding_dataloader(options, 'trn')
    embedding_val_dataloader = init_embedding_dataloader(options, 'val')

    model = init_embeddingnet(options)
    optim = init_optim(options, model)
    lr_scheduler = init_lr_scheduler(options, optim)
    res = train(opt=options,
                trn_dataloader=embedding_trn_dataloader,
                val_dataloader=embedding_val_dataloader,
                model=model,
                optim=optim,
                lr_scheduler=lr_scheduler)
    best_state, best_acc, train_loss, train_acc, val_loss, val_acc = res


if __name__ == '__main__':
    main()
