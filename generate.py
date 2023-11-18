from models.DCGAN import DCGAN
from utils.utils import save_image, save_sample_image
from argparse import ArgumentParser
import torch
from dataset import create_dataset
from utils.utils import load_yaml_with_omegaconf
from utils.utils import get_mean_std_from_batch, get_noise_from_mean_std


def parse_option():
    parser = ArgumentParser()
    parser.add_argument("-cp", "--checkpoint_path", type=str, default="./checkpoints/model.pth", help="checkpoint path")
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--config", type=str, default="./config.yml", help="config path")

    # generator param
    parser.add_argument("-bs", "--batch_size", type=int, default=16, help="the number of generated images")
    parser.add_argument("--steps", type=int, default=1, help="the times of applying model")

    # save image param
    parser.add_argument("--nrow", type=int, default=4, help="the number of images in a row")
    parser.add_argument("--show", default=False, action="store_true", help="show image")
    parser.add_argument("-sp", "--image_save_path", type=str, default=None, help="image save path")
    parser.add_argument("--to_grayscale", default=False, action="store_true", help="convert image to grayscale")

    args = parser.parse_args()
    return args


@torch.no_grad()
def generate(args):
    device = torch.device(args.device)
    conf = load_yaml_with_omegaconf(args.config)

    # load model
    cp = torch.load(args.checkpoint_path)
    model = DCGAN(**conf.model)
    model.load_state_dict(cp)
    model.to(device)
    model = model.eval()

    # load dataset, and get a batch of images to calculate mean and std
    loader = create_dataset(**conf.dataset)
    x, _ = next(iter(loader))
    x = x.to(device)

    # get noise
    mean_std = get_mean_std_from_batch(x)
    z = get_noise_from_mean_std(args.batch_size, *mean_std)
    z = z.to(device)

    if args.steps == 1:
        z = model(z)
        save_image(z, nrow=args.nrow, show=args.show, path=args.image_save_path, to_grayscale=args.to_grayscale)
    elif args.steps > 1:
        outputs = []
        for i in range(args.steps):
            z = model(z)
            outputs.append(z)
        outputs = torch.stack(outputs, dim=1)
        save_sample_image(outputs, show=args.show, path=args.image_save_path, to_grayscale=args.to_grayscale)
    else:
        raise ValueError("steps must be greater than 0")


if __name__ == '__main__':
    args = parse_option()
    generate(args)
