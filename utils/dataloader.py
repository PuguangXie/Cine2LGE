import os
import torch
import numpy as np
import nibabel as nib
import pandas as pd
from torch.utils.data import Dataset
from utils.tools import Normalization, ImageTransform
from scipy.interpolate import interp1d

def resample_to_n_frames(data, num_frames=25):
    """Resample the 3rd axis (frames) of data to num_frames."""
    original_depth = data.shape[2]

    # Upsample if needed
    if original_depth <= num_frames:
        x = np.arange(original_depth)
        f = interp1d(x, data, axis=2, kind='linear', fill_value='extrapolate')
        new_x = np.linspace(0, original_depth - 1, num_frames)
        resampled_data = f(new_x)
    # Downsample if needed
    else:
        step = original_depth / num_frames
        indices = (np.arange(num_frames) * step).astype(int)
        resampled_data = data[:, :, indices]

    return resampled_data

class CrossModalDataLoader(Dataset):
    def __init__(self, path, file_name, size, stage='Train'):
        """
        Args:
            path (str): Data root directory.
            file_name (str): CSV filename describing dataset.
            size (int): Cropping/resize size for images.
            stage (str): 'Train', 'Val', or 'Test' for data augmentation.
        """
        self.path = path
        self.crop_size = size
        self.stage = stage

        # DataFrame holding meta and target info
        df_path = os.path.join(self.path, file_name)
        self.img_df = pd.read_csv(df_path)

        if stage=='Train':
            self.cine_paths = [
                os.path.join(self.path, "train", f"{pid}_cine.nii.gz")
                for pid in self.img_df['PatientInfo']
            ]
            self.lge_paths = [
                os.path.join(self.path, "train", f"{pid}_lge.nii.gz")
                for pid in self.img_df['PatientInfo']
            ]
        if stage == 'Test':
            self.cine_paths = [
                os.path.join(self.path, "test", f"{pid}_cine.nii.gz")
                for pid in self.img_df['PatientInfo']
            ]
            self.lge_paths = [
                os.path.join(self.path, "test", f"{pid}_lge.nii.gz")
                for pid in self.img_df['PatientInfo']
            ]

        self.label = self.img_df['label']

        self.normalize = Normalization()
        self.image_transform = ImageTransform(self.crop_size, self.stage)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # Load NIfTI image data
        cine_raw = nib.load(self.cine_paths[index])
        lge_raw = nib.load(self.lge_paths[index])

        cine_imgs = cine_raw.get_fdata().astype(np.float32)
        lge_imgs = lge_raw.get_fdata().astype(np.float32)

        # Extract frame/slice info from meta
        cine_frame_len = int(self.img_df['cine_frame_len'][index])
        cine_one_idx = int(self.img_df['cine_slice_one_index'][index])
        lge_slice_idx = int(self.img_df['lge_slice_index'][index])

        # Extract cine slices, then resample to target frames
        cine_slice = cine_imgs[:, :, lge_slice_idx * cine_frame_len:(lge_slice_idx + 1) * cine_frame_len]
        cine_slice_one = cine_slice[:, :, cine_one_idx: cine_one_idx + 1]
        lge_slice = lge_imgs[:, :, lge_slice_idx: lge_slice_idx + 1]

        cine_slice = resample_to_n_frames(cine_slice, 25)
        image_cine = np.concatenate([cine_slice, cine_slice_one], axis=2)
        image_cine = self.normalize(image_cine, mode='Max_Min')
        lge_slice = self.normalize(lge_slice, mode='Max_Min')
        lge_slice = (lge_slice - 0.5) / 0.5  # Normalize to [-1, 1]

        # Concatenate for augmentation
        image = np.concatenate([image_cine, lge_slice], axis=2)
        image_trans = self.image_transform(image)

        # Separate cine & lge after augmentation
        cines_img = image_trans[:-1, :, :]
        lge_img = image_trans[-1, :, :].unsqueeze(0)

        # Label
        label_val = float(self.label.iloc[index])
        label_tensor = torch.tensor(label_val, dtype=torch.float32)

        # Final normalization for model input
        cines_img = (cines_img - 0.5) / 0.5  # Normalize to [-1, 1]

        # Return:
        # (cine stack without last slice, last cine frame, lge image, label)
        return (
            cines_img[:-1, :, :],  # [frames-1, H, W]
            cines_img[-1, :, :].unsqueeze(0),  # [1, H, W]
            lge_img,  # [1, H, W]
            label_tensor
        )
