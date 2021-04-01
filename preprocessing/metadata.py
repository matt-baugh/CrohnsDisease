import os
import pandas as pd
import numpy as np
from enum import Enum
import SimpleITK as sitk


class SeriesTypes(Enum):
    AXIAL = 'Axial T2'
    CORONAL = 'Coronal T2'
    AXIAL_POSTCON = 'Axial Postcon'
    AXIAL_POSTCON_UPPER = 'Axial Postcon Upper'
    AXIAL_POSTCON_LOWER = 'Axial Postcon Lower'


class Patient:
    def __init__(self, group, index):
        self.group = group
        self.index = index

        self.axial = None
        self.coronal = None
        self.axial_postcon = None

        self.axial_image = None
        self.coronal_image = None
        self.axial_postcon_image = None

        self.severity = None
        self.ileum = None

        self.ileum_physical = None
        self.ileum_box_size = None

        self.axial_postcon_split = False
        self.axial_postcon_upper = None
        self.axial_postcon_lower = None
        self.axial_postcon_upper_image = None
        self.axial_postcon_lower_image = None

    def get_id(self):
        return self.group + str(self.index)

    def set_paths(self, axial, coronal='', axial_postcon=''):
        self.axial = axial
        self.coronal = coronal
        if type(axial_postcon) is list:
            self.axial_postcon_split = True
            self.axial_postcon_upper = axial_postcon[0]
            self.axial_postcon_lower = axial_postcon[1]
        else:
            self.axial_postcon = axial_postcon

    def set_severity(self, severity):
        self.severity = severity

    def set_images(self, axial_image=None, coronal_image=None, axial_postcon_image=None):
        if axial_image:
            self.axial_image = axial_image
        if coronal_image:
            self.coronal_image = coronal_image
        if axial_postcon_image:
            self.axial_postcon_image = axial_postcon_image

    def set_ileum_coordinates(self, coords):
        self.ileum = coords

    def load_image_data(self, axial=True, coronal=False, axial_postcon=False):
        if axial:
            if os.path.isfile(self.axial):
                self.axial_image = sitk.ReadImage(self.axial)
            else:
                print(f'Patient {self.get_id()} is missing Axial T2 image: {self.axial}')
        if coronal:
            if os.path.isfile(self.coronal):
                orig_coronal_image = sitk.ReadImage(self.coronal)
                orig_size = orig_coronal_image.GetSize()
                orig_spacing = orig_coronal_image.GetSpacing()
                orig_direction = orig_coronal_image.GetDirection()

                # Need to change the viewing direction of the coronal scan, to match the axial scans

                new_origin = orig_coronal_image.TransformIndexToPhysicalPoint([0, orig_size[1] - 1, 0])
                rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
                orig_dir_np = np.reshape(np.array(orig_direction), (3, 3))
                new_direction = (orig_dir_np @ rotation_matrix).flatten()

                self.coronal_image = sitk.Resample(orig_coronal_image,
                                                   [orig_size[0], orig_size[2], orig_size[1]],
                                                   sitk.Transform(),
                                                   sitk.sitkLinear,
                                                   new_origin,
                                                   (orig_spacing[0], orig_spacing[2], orig_spacing[1]),
                                                   new_direction)
            else:
                print(f'Patient {self.get_id()} is missing Coronal T2 image: {self.coronal}')
        if axial_postcon:
            if not self.axial_postcon_split and os.path.isfile(self.axial_postcon):

                self.axial_postcon_image = sitk.ReadImage(self.axial_postcon)

            elif self.axial_postcon_split and os.path.isfile(self.axial_postcon_upper)\
                    and os.path.isfile(self.axial_postcon_lower):

                self.axial_postcon_upper_image = sitk.ReadImage(self.axial_postcon_upper)
                self.axial_postcon_lower_image = sitk.ReadImage(self.axial_postcon_lower)

            else:
                print(f'Patient {self.get_id()} is missing Axial Postcon image: {self.axial_postcon}')

    def __str__(self):
        return f'{self.get_id()}: {self.axial}, {self.coronal}, {self.axial_postcon}'


class Metadata:
    def form_path(self, patient, series_type):
        folder = patient.group
        if self.dataset_tag:
            folder += self.dataset_tag
        path = os.path.join(self.data_path, folder, f'{patient.get_id()} {series_type}{self.dataset_tag}')
        for ext in self.data_extensions:
            full_path = f'{path}.{ext}'
            if os.path.isfile(full_path):
                return full_path

        # Some Axial Postcontrast scans are split into a upper and lower portion
        if series_type is SeriesTypes.AXIAL_POSTCON.value:
            upper_path = os.path.join(self.data_path, folder, f'{patient.get_id()} {SeriesTypes.AXIAL_POSTCON_UPPER.value}{self.dataset_tag}')
            upper_found = False
            full_upper_path = None
            for ext in self.data_extensions:
                full_upper_path = f'{upper_path}.{ext}'
                if os.path.isfile(full_upper_path):
                    upper_found = True
                    break

            if not upper_found:
                return -1

            lower_path = os.path.join(self.data_path, folder, f'{patient.get_id()} {SeriesTypes.AXIAL_POSTCON_LOWER.value}{self.dataset_tag}')
            for ext in self.data_extensions:
                full_lower_path = f'{lower_path}.{ext}'
                if os.path.isfile(full_lower_path):
                    return [full_upper_path, full_lower_path]
        return -1

    def file_list(self, patients):
        for patient in patients:
            axial = self.form_path(patient, SeriesTypes.AXIAL.value)
            coronal = self.form_path(patient, SeriesTypes.CORONAL.value)
            axial_postcon = self.form_path(patient, SeriesTypes.AXIAL_POSTCON.value)
            patient.set_paths(axial, coronal, axial_postcon)
        return patients

    def label_set(self, patients, labels):
        for patient in patients:
            patient_labels = labels.loc[labels['Case ID'] == patient.get_id()]
            patient.set_severity(np.array(patient_labels['Terminal ileal Inflammation level'])[0])
            patient.set_ileum_coordinates(np.array(patient_labels[['coronal', 'sagittal', 'axial']])[0])
        return patients

    def __init__(self, data_path, label_path, abnormal_cases, healthy_cases, dataset_tag=''):
        print('Forming metadata')
        self.data_path = data_path
        self.data_extensions = ['nii', 'nii.gz']
        self.dataset_tag = dataset_tag

        abnormal_patients = [Patient('A', i + 1) for i in abnormal_cases]
        healthy_patients = [Patient('I', i + 1) for i in healthy_cases]

        self.patients = self.file_list(abnormal_patients + healthy_patients)

        ileum_labels = pd.read_csv(os.path.join(label_path, 'terminal_ileum'), delimiter='\t', header=0)
        self.patients = self.label_set(self.patients, ileum_labels)
