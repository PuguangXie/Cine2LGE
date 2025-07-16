import os
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from accelerate import Accelerator
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from models import DiT_models
from diffusion import create_diffusion
from utils.dataloader import CrossModalDataLoader
from vivit import ViViT

def set_requires_grad(model, requires_grad: bool = True):
    """
    Sets the requires_grad attribute for all model parameters.
    """
    for param in model.parameters():
        param.requires_grad = requires_grad

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        raise EnvironmentError("Training currently requires at least one GPU.")

def main(args):
    # Environment preparation
    accelerator = Accelerator()
    device = accelerator.device if torch.cuda.is_available() else torch.device("cpu")
    seed_everything(2023)
    scaler = GradScaler()

    # Load configuration
    config_path = os.path.join('..', args.config)
    cfg = OmegaConf.load(config_path)
    cfg.data.dataset.path = os.path.join('..', 'data')

    # Model Preparation
    model_cls = DiT_models[args.model]
    model = model_cls(input_size=args.image_size, learn_sigma=True).to(device)
    cine_encoder = ViViT(args.image_size, 16, 2, 25).to(device)
    diffusion = create_diffusion(timestep_respacing="", learn_sigma=True)

    # Optimizer and loss function
    optimizer = torch.optim.AdamW(
        [{'params': model.parameters(), 'lr': 1e-4},
         {'params': cine_encoder.parameters(), 'lr': 1e-4}]
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    # Data preparation
    train_dataset = CrossModalDataLoader(
        file_name='train_group.csv', stage='Train',
        path=cfg.data.dataset.path, size=160
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=8, drop_last=True
    )

    # Create checkpoint directory
    ckpt_dir = os.path.join('logs', args.model_name, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        cine_encoder.train()
        total_loss_mse = 0.0
        total_loss_cat = 0.0

        progress_bar = tqdm(train_loader, unit="batch", mininterval=0.3)
        for batch in progress_bar:
            cine_img, cine_one_img, lge_img, label = batch
            progress_bar.set_description(f"Epoch {epoch+1}/{args.epochs}")

            # Move tensors to device, ensuring correct dtype
            cine_img = cine_img.to(device)
            cine_one_img = cine_one_img.to(device)
            lge_img = lge_img.to(device)
            label = label.to(device, dtype=torch.float32)

            x = lge_img
            y, y_cat = cine_encoder(cine_img.unsqueeze(2))
            x_cat = torch.cat([cine_one_img, cine_img], dim=1)
            t = torch.randint(0, diffusion.num_timesteps, (lge_img.size(0),), device=device)

            loss_cat = criterion(y_cat.squeeze(1), label)
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, x_cat, t, model_kwargs=model_kwargs)
            loss = loss_dict["loss"].mean()
            loss_all = loss_cat + loss

            optimizer.zero_grad()
            scaler.scale(loss_all).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss_mse += loss.item()
            total_loss_cat += loss_cat.item()

        print(f"End of epoch {epoch+1} | Loss MSE: {total_loss_mse:.5f} | Loss Cat: {total_loss_cat:.5f}")

        # Checkpoint saving
        if (epoch + 1) % 2 == 0 or (epoch + 1) == args.epochs:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"model_epoch_{epoch+1}.pth"))
            torch.save(cine_encoder.state_dict(), os.path.join(ckpt_dir, f"cine_encoder_epoch_{epoch+1}.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Cine2LGE Model")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-B/2")
    parser.add_argument("--image_size", type=int, default=160, help="Input image size.")
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--model_name", type=str, default="cine2lge")
    parser.add_argument("--config", type=str, default='config.yaml', help="Path to config yaml.")
    args = parser.parse_args()

    main(args)
