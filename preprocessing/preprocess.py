import SimpleITK as sitk
import matplotlib.pyplot as plt

import numpy as np
import math

def show_data(data, sl, name):
    fig = plt.figure(figsize=(18, 18))
    fig.set_size_inches(15, 10)
    columns = 8
    rows = math.ceil(len(data) / columns)
    for i, image in enumerate(data):
        fig.add_subplot(rows, columns, i + 1)
        nda = sitk.GetArrayFromImage(image) / 255
        nda = nda.astype(np.float32)

        plt.imshow(nda[sl], cmap='gray')
    plt.savefig(f'images/{name}.png')

def image_physical_center(image):
    return np.array(image.TransformContinuousIndexToPhysicalPoint(np.array(image.GetSize())/2.0))

class Preprocessor:
    def generate_reference_volume(self, patient):
        # Physical image size corresponds to the largest physical size in the training set, or any other arbitrary size.
        reference_physical_size = patient.ileum_box_size

        # Create the reference image with a zero origin, identity direction cosine matrix and dimension
        reference_origin = patient.ileum_physical - reference_physical_size / 2
        reference_direction = np.identity(self.dimension).flatten()
        reference_spacing = [phys_sz / sz for sz, phys_sz in zip(self.constant_volume_size, reference_physical_size)]

        reference_image = sitk.Image(self.constant_volume_size, patient.axial_image.GetPixelIDValue())
        reference_image.SetOrigin(reference_origin)
        reference_image.SetSpacing(reference_spacing)
        reference_image.SetDirection(reference_direction)

        return reference_image

    def __init__(self, constant_volume_size=[256, 128, 64]):
        self.constant_volume_size = constant_volume_size

    def process(self, patients, ileum_crop=False, region_grow_crop=False, statistical_region_crop=False):
        print('Preprocessing...')
        self.dimension = patients[0].axial_image.GetDimension()

        # Patient specific cropping to Terminal Ileum (semi-automatic preprocessing)
        if ileum_crop:
            print('Cropping to Ileum...')
            for patient in patients:
                # Provided coordinates are [coronal, sagittal, axial]
                # convert to [sagittal, coronal, axial]
                parsed_ileum = np.array([patient.ileum[1], patient.ileum[0], patient.ileum[2]])
                patient.ileum_physical = patient.axial_image.TransformContinuousIndexToPhysicalPoint(parsed_ileum * 1.0)
                patient.ileum_box_size = np.array([80, 80, 112])


        # Population specific cropping (fully-automatic preprocessing)
        elif region_grow_crop:
            # First crop to patient (also to determine rough patient dimensions)
            for patient in patients:
                patient.set_images(self.region_grow_crop(patient))
            # Then crop to proportional generic region guaranteed to contain Terminal Ileum
            if statistical_region_crop:
                # Proportional generic region derived externally (format: [sag, cor, ax])
                normalised_ilea_mean = np.array([-0.192, -0.1706, -0.1114])
                normalised_ilea_box_size = np.array([0.289, 0.307483, 0.4804149]) * 1.1

                for patient in patients:
                    image_phys_size = np.array(patient.axial_image.GetSize()) * patient.axial_image.GetSpacing()
                    patient.ileum_physical = image_physical_center(patient.axial_image) + normalised_ilea_mean * image_phys_size
                    patient.ileum_box_size = normalised_ilea_box_size * image_phys_size

        # print('Showing data...')
        # show_data([p.axial_image for p in patients], 13, 'cropped')
        # [sitk.WriteImage(patients[i].axial_image, f'images/patient_{i}.nii', True) for i in range(3)]

        # Resample
        print(f'Resampling volumes to {self.constant_volume_size}')
        for patient in patients:
            # Make empty image with same metadata
            reference_volume = self.generate_reference_volume(patient)

            patient.set_images(axial_image=sitk.Resample(patient.axial_image, reference_volume))

            if patient.coronal_image is not None:
                patient.set_images(coronal_image=sitk.Resample(patient.coronal_image, reference_volume))

            if patient.axial_postcon_image is not None:
                patient.set_images(axial_postcon_image=sitk.Resample(patient.axial_postcon_image, reference_volume))

        # show_data([p.axial_image for p in patients], 13, 'resample')

        return patients

    def region_grow_crop(self, patient):
        image = patient.axial_image

        # Define parameters and seed points of region growing
        inside_value = 20
        outside_value = 255
        seed = (int(image.GetSize()[0]/2), int(image.GetSize()[1]/2), int(image.GetSize()[2]/2))
        perturbations = [-5, 5]
        seeds = [seed]
        seeds += [(seed[0], seed[1], seed[2] + p) for p in perturbations]
        seeds += [(seed[0], seed[1] + p, seed[2]) for p in perturbations]

        # Apply region growing
        seg_explicit_thresholds = sitk.ConnectedThreshold(image, seedList=seeds,
                                                          lower=inside_value, upper=outside_value)
        overlay = sitk.LabelOverlay(image, seg_explicit_thresholds)

        # Label grown-region as 1
        label = 1
        label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
        label_shape_filter.Execute(seg_explicit_thresholds)

        # Crop image to bounding box of grown region
        bounding_box = label_shape_filter.GetBoundingBox(label)
        cropped = sitk.RegionOfInterest(image, bounding_box[int(len(bounding_box)/2):], bounding_box[0:int(len(bounding_box)/2)])

        # Calculate and print ileum location relative to cropped image
        # Used for manual average calculation, which is applied as hardcoded value in statisical_region_crop mode

        # Calculate cropped image physical center and size
        crop_center = image_physical_center(cropped)
        crop_physical_quadrant_size = np.array([spc * sz for spc, sz in zip(cropped.GetSpacing(), cropped.GetSize())]) / 2.0
        # Calculate ileum relative position inside cropped image
        ileum = [patient.ileum[1], patient.ileum[0], patient.ileum[2]]
        physical_ileum_coords = image.TransformContinuousIndexToPhysicalPoint(np.array(ileum) * 1.0)
        ileum_prop = (np.array(physical_ileum_coords) - crop_center) / crop_physical_quadrant_size
        str_ileum_prop = [str(x) for x in ileum_prop]
        str_ileum_prop = ('\t').join(str_ileum_prop)

        # Metrics for ilea distribution
        print(f'{patient.get_id()}\t{patient.group}\t{patient.severity}\t{str_ileum_prop}')
        # print(f'{patient.get_id()}\t{(np.array(physical_ileum_coords) - crop_center) / crop_physical_quadrant_size).join('\t')}')
        # print(patient.ileum, cropped.TransformPhysicalPointToIndex(physical_ileum_coords))
        return cropped
