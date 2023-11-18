from dataset import create_dataset
from utils.utils import load_yaml_with_omegaconf
from models.DCGAN import DCGAN
from utils.engine import train_one_epoch
from argparse import ArgumentParser
from torch.optim import Adam
import torch


def main(args, conf):
    print(conf)

    device = torch.device(args.device)
    data_loader = create_dataset(**conf.dataset)
    start_epoch = 1

    model = DCGAN(**conf.model).to(device)
    model_copy = DCGAN(**conf.model).to(device).requires_grad_(False)
    optimizer = Adam(model.parameters(), **conf.optimizer)

    for epoch in range(start_epoch, args.epochs + 1):
        loss = train_one_epoch(model, model_copy, optimizer, data_loader, device, epoch, **conf.loss)
        torch.save(model.state_dict(), 'model.pth')


if __name__ == '__main__':
    paser = ArgumentParser()
    paser.add_argument('-c', '--config', type=str, default='./config.yml')
    paser.add_argument('-d', '--device', type=str, default='cuda')
    paser.add_argument('-e', '--epochs', type=int, default=1000)
    args = paser.parse_args()

    conf = load_yaml_with_omegaconf(args.config)
    main(args, conf)
