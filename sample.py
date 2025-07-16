import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from utils.dataloader import CrossModalDataLoader
from utils.tools import metrics
from diffusion import create_diffusion
from models import DiT_models
from vivit import ViViT
import argparse

def sample(args, epoch, model, cine_encoder, device):
    """
    Test the model on the independent test set and report evaluation metrics.
    """
    dataset = CrossModalDataLoader(
        file_name='test.csv',
        stage='Test',
        path=args.data_path,
        size=args.image_size
    )

    dataloader = DataLoader(
        dataset,
        batch_size=100,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )

    with torch.no_grad():
        psnr_list, ssim_list, mse_list = [], [], []
        probabilities_all, labels_all = [], []

        # Create diffusion instance only once outside the loop for efficiency
        diffusion = create_diffusion(str(args.num_sampling_steps))
        generator = torch.Generator(device).manual_seed(2023)

        for batch in dataloader:
            # Unpack batch data;
            cine_img, cine_one_img, lge_img, label = batch
            cine_img = cine_img.to(device)
            cine_one_img = cine_one_img.to(device)
            lge_img = lge_img.to(device)
            label = label.to(device)
            lge_img_original = lge_img.clone()

            # Sample random noise z
            z = torch.randn_like(lge_img, generator=generator, device=device)
            y, y_cat = cine_encoder(cine_img.unsqueeze(2))
            probabilities = torch.sigmoid(y_cat).cpu().numpy().flatten()
            labels_np = label.cpu().detach().numpy().flatten()
            probabilities_all.extend(probabilities)
            labels_all.extend(labels_np)

            model_kwargs = dict(y=y)
            x_cat = torch.cat([cine_one_img, cine_img], dim=1)

            # Generate fake images using the diffusion process
            gen_image = diffusion.p_sample_loop(
                model.forward,
                z.shape,
                z,
                x_cat,
                clip_denoised=False,
                progress=True,
                device=device,
                model_kwargs=model_kwargs
            )

            # Rescale to [0,1] for evaluation
            gen_image = torch.clamp((gen_image + 1) / 2.0, 0.0, 1.0)
            lge_img_original = torch.clamp((lge_img_original + 1) / 2.0, 0.0, 1.0)

            # Compute evaluation metrics
            psnr, ssim, mse = metrics(gen_image, lge_img_original)
            psnr_list += psnr
            ssim_list += ssim
            mse_list += mse

        # Save metrics to file
        metrics_df = pd.DataFrame({
            'PSNR': psnr_list,
            'SSIM': ssim_list,
            'MSE': mse_list,
        })
        metrics_df.to_csv(
            os.path.join(args.results_dir, f'test_metrics_epoch_{epoch}.csv'),
            index=False
        )

        return psnr_list, ssim_list, mse_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on.")
    parser.add_argument("--model_name", type=str, default="cine2lge")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-B/2")
    parser.add_argument("--image_size", type=int, default=160)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--state_dict_model", type=str, default='./checkpoints/model.pth', help="Path to DiT model checkpoint.")
    parser.add_argument("--state_dict_cine", type=str, default='./checkpoints/cine_encoder.pth', help="Path to cine encoder checkpoint.")
    parser.add_argument("--cfg_scale", type=float, default=1)
    parser.add_argument("--num_sampling_steps", type=int, default=250)
    args = parser.parse_args()

    # Load config and set device
    cfg = OmegaConf.load(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model and weights
    model = DiT_models[args.model](
        input_size=args.image_size,
        learn_sigma=True
    ).to(device)
    cine_encoder = ViViT(args.image_size, 16, 2, 25).to(device)
    model.load_state_dict(torch.load(args.state_dict_model))
    cine_encoder.load_state_dict(torch.load(args.state_dict_cine))
    model.eval()
    cine_encoder.eval()

    os.makedirs(args.results_dir, exist_ok=True)
    # Run the test process
    sample(args, epoch=2000, model=model, cine_encoder=cine_encoder, device=device)