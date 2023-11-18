import torch
from tqdm import tqdm
from torch.nn import functional as F
from utils.utils import get_mean_std_from_batch, get_noise_from_mean_std


def train_one_epoch(f, f_copy, optimizer, data_loader, device, epoch,
                    lambda_rec=1, lambda_idem=1, lambda_tight=1, a=1):
    f.train()
    total_loss, total_num = 0., 0

    data_loader = tqdm(data_loader, dynamic_ncols=True, colour="#ff924a")
    for x, _ in data_loader:
        batch_size = x.shape[0]
        x = x.to(device)

        # noise
        mean_std = get_mean_std_from_batch(x)
        z = get_noise_from_mean_std(batch_size, *mean_std)
        z = z.to(device, memory_format=torch.contiguous_format)

        f_copy.load_state_dict(f.state_dict())
        fx = f(x)
        fz = f(z)
        f_z = fz.detach()
        ff_z = f(f_z)
        f_fz = f_copy(fz)

        # loss
        loss_rec = F.l1_loss(fx, x, reduction='none').reshape(batch_size, -1).mean(dim=1)
        loss_idem = F.l1_loss(f_fz, fz, reduction='mean')
        loss_tight = -F.l1_loss(ff_z, f_z, reduction='none').reshape(batch_size, -1).mean(dim=1)
        # smooth
        smooth_loss_tight = F.tanh(loss_tight / (a * loss_rec)) * a * loss_rec
        loss_rec = loss_rec.mean()
        smooth_loss_tight = smooth_loss_tight.mean()

        loss = lambda_rec * loss_rec + lambda_idem * loss_idem + lambda_tight * smooth_loss_tight

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        total_num += batch_size

        data_loader.set_description(f"Epoch: {epoch}")
        data_loader.set_postfix(ordered_dict={
            "train_loss": total_loss / total_num,
        })

    return total_loss / total_num
