from mri_dataset import MRIDataset
from torch.utils.data import DataLoader

dataset_path = '/vol/bitbucket/mb4617/MRI_Crohns/numpy_datasets/ti_imb/axial_t2_only_train_fold0.npz'

record_shape = (37, 99, 99)
feature_shape = (31, 87, 87)

dataset = MRIDataset(dataset_path, train=True, out_dims=feature_shape)

dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

for i_batch, (x, y) in enumerate(dataloader):
    print(i_batch)
    print(x.shape)
    print(x[0])
    break