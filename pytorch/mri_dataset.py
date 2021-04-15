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


class RandomModalityShift:
    def __init__(self, out_shape):
        ## Must be even number less than input
        self.out_shape = out_shape

    def _shift_single_channel(self, channel):
        depth, height, width = self.out_shape
        d, h, w = channel.shape

        z_diff = d - depth
        y_diff = h - height
        x_diff = w - width

        k = torch.randint(0, z_diff + 1, size=(1,)).item()
        j = torch.randint(0, y_diff + 1, size=(1,)).item()
        i = torch.randint(0, x_diff + 1, size=(1,)).item()

        return channel[k: k + depth, j: j + height, i: i + width]

    def __call__(self, x):
        c, d, h, w = x.shape
        depth, height, width = self.out_shape

        # Do nothing if only single modality
        if c == 1:
            return x

        axial = x[0]

        z_d = (d - depth) // 2
        y_d = (h - height) // 2
        x_d = (w - width) // 2

        axial_cropped = axial[z_d: z_d + depth, y_d: y_d + height, x_d: x_d + width]

        return torch.stack([axial_cropped, *[self._shift_single_channel(ch) for ch in x[1:]]])


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
    def __init__(self, dataset_path, train, in_dims, out_dims, input_features, transforms=None, preprocess=False):
        ## Load dataset
        np_dataset_file = np.load(dataset_path)

        self.out_dims = out_dims
        self.train = train
        self.preprocess = preprocess

        normalize_3d = T.Lambda(
            lambda x: (x - torch.mean(x, dim=[1, 2, 3], keepdim=True)) /
                       torch.std(x, dim=[1, 2, 3], keepdim=True)
        )

        inter_dim = np.ceil((np.array(in_dims) + np.array(out_dims)) / 2).astype(int)

        ## Define transforms to be applied to data
        if transforms is not None:
            self.transforms = transforms
        elif train:
            self.transforms = T.Compose([
                RandomModalityShift(inter_dim),
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
