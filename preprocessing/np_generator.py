import os
import SimpleITK as sitk
import random
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
import numpy as np


def numpy_dataset_name(set, suffix=''):
    return f'{suffix}_{set}.npz'


def get(ls, ixs):
    return [ls[i] for i in ixs]


class NumpyGenerator:
    def __init__(self, out_path, suffix):
        self.out_path = out_path
        self.suffix = suffix

        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)

        self.write_log(f'Dataset created: {str(datetime.now())}')

    def write_log(self, line):
        with open(os.path.join(self.out_path, 'METADATA'), 'a') as levels:
            levels.write(f'{line}\n')

    def _generate_dataset(self, patients, set='train', fold=''):
        dataset_filename = os.path.join(self.out_path,
                                        numpy_dataset_name(f'{set}_{fold}', self.suffix))

        abnormal = [p.index for p in patients if p.group == 'A']
        healthy = [p.index for p in patients if p.group == 'I']
        self.write_log(f'{set} set, fold {fold}:')
        self.write_log(f'A - {abnormal}')
        self.write_log(f'I - {healthy}')
        self.write_log([p.severity for p in patients])

        axial_t2 = np.stack([sitk.GetArrayFromImage(p.axial_image)
                             for p in patients])
        coronal_t2 = np.stack([sitk.GetArrayFromImage(p.coronal_image)
                               for p in patients])
        axial_pc = np.stack([sitk.GetArrayFromImage(p.axial_postcon_image)
                             for p in patients])

        self.write_log(f'fold dataset dimensions {axial_t2.shape}')
        label = np.array([p.severity for p in patients])
        index = np.array([p.index for p in patients])

        np.savez(dataset_filename, axial_t2=axial_t2, coronal_t2=coronal_t2, axial_pc=axial_pc,
                 label=label, index=index)

    def generate_cross_folds(self, k, patients):
        random.shuffle(patients)

        self.write_log(f'Volume size: {sitk.GetArrayFromImage(patients[0].axial_image).shape}')

        if k == 1:
            print('Creating single dataset')
            self._generate_dataset(patients, set='all', fold='data')
            return

        y = [patient.severity for patient in patients]
        skf = StratifiedKFold(n_splits=k)
        for i, (train, test) in enumerate(skf.split(patients, y)):
            patients_train = get(patients, train)
            print('Creating train data...')
            self._generate_dataset(patients_train, set='train', fold=f'fold{i}')

            patients_test = get(patients, test)
            print('Creating test data...')
            self._generate_dataset(patients_test, set='test', fold=f'fold{i}')

