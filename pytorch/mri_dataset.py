import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, TensorDataset
import numpy as np


class FastRotate:

    def __init__(self, std):
        self.std = std

    def __call__(self, x):
        angle = np.random.normal(loc=0, scale=self.std)

        _, _, height, width = x.shape
        corner_angle = np.arctan(height / width)
        rad_angle = np.radians(np.abs(angle))

        distance_to_top_corner = np.hypot(height, width) * 0.5 * np.sin(corner_angle + rad_angle)
        pad_amount = int(np.ceil(distance_to_top_corner - height // 2))

        x = TF.pad(x, pad_amount, padding_mode='edge')

        x = TF.rotate(x, angle, interpolation=T.InterpolationMode.BILINEAR)
        return TF.center_crop(x, [height, width])


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


INDEX_TO_MODALITY = {
    0: 'axial_t2',
    1: 'coronal_t2',
    2: 'axial_pc'
}


class MRIDataset(Dataset):
    def __init__(self, dataset_path, train, out_dims, input_features, transforms=None, preprocess=False):
        ## Load dataset
        np_dataset_file = np.load(dataset_path)

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
                FastRotate(4),
                T.RandomHorizontalFlip(),
                T.Lambda(_random_crop_gen(out_dims)),
                T.Lambda(lambda x: x + 5 * torch.randn_like(x)),
                normalize_3d
            ])
        else:
            self.transforms = T.Compose([
                T.Lambda(_center_crop_gen(out_dims)),
                normalize_3d
            ])

        self.chosen_data = [INDEX_TO_MODALITY[i] for i in range(len(INDEX_TO_MODALITY)) if input_features[i]]
        print(self.chosen_data)
        input_data = torch.stack([torch.from_numpy(np_dataset_file[m]).float() for m in self.chosen_data], 1)

        if preprocess:
            input_data = torch.stack([self.transforms(d) for d in input_data])

        self.data = TensorDataset(input_data,
                                  torch.from_numpy(np_dataset_file['label']))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_data, label = self.data[idx]

        if not self.preprocess and self.transforms:
            sample_data = self.transforms(sample_data)

        return sample_data, label
