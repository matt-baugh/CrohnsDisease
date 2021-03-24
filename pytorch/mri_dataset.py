import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, TensorDataset
from scipy.ndimage import rotate
import numpy as np


def _crop_out_edges(x):
    return x[:, 2:-2, 4:-4, 4:-4]


def _random_rotate(x):
    angle = np.random.normal(loc=0, scale=4)
    rotated_np = rotate(x, angle, axes=(2, 3), reshape=False, order=5, mode='nearest')
    return torch.from_numpy(rotated_np)


def _random_crop_gen(out_dims):
    final_crop = T.RandomCrop((out_dims[1], out_dims[2]))

    def _random_crop_helper(x):
        # Have to crop first dimension manually, as pytorch only does last 2
        # First tensor dimension is channel, so ignore
        dim_0_diff = x.shape[1] - out_dims[0]
        dim_0_crop = np.random.randint(dim_0_diff)
        return final_crop(x[:, dim_0_crop:dim_0_crop+out_dims[0], :, :])
    return _random_crop_helper


def _center_crop_gen(out_dims):
    final_crop = T.CenterCrop((out_dims[1], out_dims[2]))

    def _center_crop_helper(x):
        dim_0_diff = x.shape[1] - out_dims[0]
        dim_0_crop = dim_0_diff // 2
        return final_crop(x[:, dim_0_crop:dim_0_crop+out_dims[0], :, :])
    return _center_crop_helper


class MRIDataset(Dataset):
    def __init__(self, dataset_path, train, out_dims, transforms=None, preprocess=False):
        ## Load dataset
        np_dataset_file = np.load(dataset_path)

        axial_data = torch.from_numpy(np_dataset_file['axial_t2']).float()

        self.out_dims = out_dims
        self.train = train
        self.preprocess = preprocess

        normalize_3d = T.Lambda(
            lambda x: (x - torch.mean(x, dim=[1, 2, 3], keepdim=True)) /
                       torch.std(x, dim=[1, 2, 3], keepdim=True)
        )

        ## Define transforms to be applied to data
        if transforms is not None:
            self.transforms = transforms
        elif train:
            self.transforms = T.Compose([
                T.Lambda(_crop_out_edges),
                T.Lambda(_random_rotate),
                T.RandomHorizontalFlip(),
                T.Lambda(_random_crop_gen(out_dims)),
                T.Lambda(lambda x: x + 0.005 * torch.randn_like(x)),
                normalize_3d
            ])
        else:
            self.transforms = T.Compose([
                T.Lambda(_center_crop_gen(out_dims)),
                normalize_3d
            ])

        if preprocess:
            axial_data = torch.stack([
                torch.squeeze(self.transforms(torch.unsqueeze(a, 0))) for a in axial_data
            ], 0)
        self.data = TensorDataset(axial_data,
                                  torch.from_numpy(np_dataset_file['label']))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        axial_data, label = self.data[idx]

        sample_data = torch.unsqueeze(axial_data, 0)

        if not self.preprocess and self.transforms:
            sample_data = self.transforms(sample_data)

        return sample_data, label
