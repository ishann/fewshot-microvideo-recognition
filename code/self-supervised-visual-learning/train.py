## TODOS
# - [ ] Add primitive logging.
# - [ ] Add checkpointing. Make sure checkpoints can be loaded and used.
# - [ ] A few arguments are hardcoded. Make them configurable from argparser.
import argparse
import ipdb
import os
from setproctitle import setproctitle

import torch
from torch.utils.data import DataLoader

from data import LabeledVideoDataset
from model import ss_slowfast_r50

def main(args):

    # setup
    setproctitle(args.exp_name)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # data and model setup
    ss_vid_data_loader = DataLoader(LabeledVideoDataset(args.device),
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers)
    model = ss_slowfast_r50()
    model = model.to(args.device)

    # optimization setup
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    losses = []

    for epoch in range(args.epochs):
        for idx, (x, y) in enumerate(ss_vid_data_loader):

            # ipdb.set_trace()
            optimizer.zero_grad()

            x, y = x.to(args.device), y.to(args.device)
            y_pred = model(x)

            loss = criterion(y_pred, y)
            losses.append(loss)
            loss.backward()
            optimizer.step()

            print(f"[{epoch}]:{idx}\t\tLoss: {loss}")

    print("Done!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "Self-supervised video finetuning.")
    parser.add_argument("--exp_name", help="Identify experiments on nvidia-smi.",
                        default="an_expt_has_no_name", required=True)
    parser.add_argument("--batch_size", help="Number of data elements per batch.",
                        default=2, required=True)
    parser.add_argument("--gpu", help="Be nice. Use only one GPU.",
                        default="1", required=True)
    parser.add_argument("--device", help="Device: cpu / gpu.",
                        default="cuda", required=True)
    parser.add_argument("--epochs", help="Number of epochs to train for.",
                        default=10, required=True)
    parser.add_argument("--num_workers", help="Number of parallel workers.",
                        default=4, required=True)

    args = parser.parse_args()

    for k, v in vars(args).items():
        print(f"{k}: {v}")

    main(args)


