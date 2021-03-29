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
        reference_physical_size = np.zeros(self.dimension)

        img = patient.axial_image
        reference_physical_size[:] = [(sz-1)*spc for sz,spc,mx in zip(img.GetSize(), img.GetSpacing(), reference_physical_size)]

        # Create the reference image with a zero origin, identity direction cosine matrix and dimension
        reference_origin = np.zeros(self.dimension)
        reference_direction = np.identity(self.dimension).flatten()
        reference_spacing = [phys_sz/(sz-1) for sz, phys_sz in zip(self.constant_volume_size, reference_physical_size)]

        reference_image = sitk.Image(self.constant_volume_size, patient.axial_image.GetPixelIDValue())
        reference_image.SetOrigin(reference_origin)
        reference_image.SetSpacing(reference_spacing)
        reference_image.SetDirection(reference_direction)

        return reference_image

    def crop_box_about_center(self, image, pixel_center, physical_crop_size):
        box_size = np.array([pcsz / vsz for vsz,pcsz in zip(image.GetSpacing(), physical_crop_size)])
        lb = np.array(pixel_center - box_size/2).astype(int)
        ub = (lb + box_size).astype(int)

        img = image[lb[0]:ub[0], lb[1]:ub[1], lb[2]:ub[2]]
        print('cropped ', img.GetSize(), ' covering ', np.array(img.GetSize()) * img.GetSpacing())
        return img

    def __init__(self, constant_volume_size=[256, 128, 64]):
        self.constant_volume_size = constant_volume_size

    def process(self, patients, ileum_crop=False, region_grow_crop=False, statistical_region_crop=False):
        print('Preprocessing...')
        self.dimension = patients[0].axial_image.GetDimension()

        # Patient specific cropping to Terminal Ileum (semi-automatic preprocessing)
        proportion_center = np.array([])
        proportion_box_size = np.array([])
        if ileum_crop:
            print('Cropping to Ileum...')
            for patient in patients:

                ax_t2 = patient.axial_image
                cor_t2 = patient.coronal_image
                ax_pc = patient.axial_postcon_image

                # Provided coordinates are [coronal, sagittal, axial]
                # convert to [sagittal, coronal, axial]
                parsed_ileum = np.array([patient.ileum[1], patient.ileum[0], patient.ileum[2]])
                parsed_ileum_physical = ax_t2.TransformContinuousIndexToPhysicalPoint(parsed_ileum * 1.0)
                physical_crop_size = np.array([80, 80, 112])
                # ileum_from_center = parsed_ileum_physical - image_physical_center(ax_t2)

                patient.set_images(axial_image=self.crop_box_about_center(ax_t2, parsed_ileum, physical_crop_size))

                if cor_t2:
                    # coronal_ileum_phys = image_physical_center(cor_t2) + ileum_from_center
                    coronal_index = cor_t2.TransformPhysicalPointToContinuousIndex(parsed_ileum_physical) #coronal_ileum_phys)
                    patient.set_images(coronal_image=self.crop_box_about_center(cor_t2, coronal_index, physical_crop_size))

                if ax_pc:
                    # axial_pc_ileum_phys = image_physical_center(ax_pc) + ileum_from_center
                    axial_pc_index = ax_pc.TransformPhysicalPointToContinuousIndex(parsed_ileum_physical) #axial_pc_ileum_phys)
                    patient.set_images(axial_postcon_image=self.crop_box_about_center(ax_pc, axial_pc_index, physical_crop_size))

                print()

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
                    ilea_mean = image_physical_center(patient.axial_image) + normalised_ilea_mean * patient.axial_image.GetSize()
                    ilea_box_size = normalised_ilea_box_size * patient.axial_image.GetSize() * patient.axial_image.GetSpacing()
                    pixel_ilea_mean = patient.axial_image.TransformPhysicalPointToIndex(ilea_mean)
                    patient.set_images(self.crop_box_about_center(patient.axial_image, pixel_ilea_mean, ilea_box_size))

        # print('Showing data...')
        # show_data([p.axial_image for p in patients], 13, 'cropped')
        # [sitk.WriteImage(patients[i].axial_image, f'images/patient_{i}.nii', True) for i in range(3)]

        # Resample
        print(f'Resampling volumes to {self.constant_volume_size}')
        for patient in patients:
            # Make empty image with same metadata
            reference_volume = self.generate_reference_volume(patient)
            reference_center = image_physical_center(reference_volume)

            patient.set_images(axial_image=self.resample(patient, reference_volume, reference_center))
        # show_data([p.axial_image for p in patients], 13, 'resample')

        return patients

    def resample(self, patient, reference_volume, reference_center):
        img = patient.axial_image

        # Transform which maps from the reference_image to the current img with the translation mapping the image
        # origins to each other.
        transform = sitk.TranslationTransform(self.dimension)
        transform.SetOffset(np.array(img.GetOrigin()) - reference_volume.GetOrigin())
        # transform.SetMatrix(img.GetDirection())
        centered_transform = sitk.Transform(transform)

        # Modify the transformation to align the centers of the original and reference image instead of their origins.
        centering_transform = sitk.TranslationTransform(self.dimension)
        img_center = image_physical_center(img)
        centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))

        combined_transform = sitk.CompositeTransform([centered_transform, centering_transform])

        # Using the linear interpolator as these are intensity images
        return sitk.Resample(img, reference_volume, combined_transform, sitk.sitkLinear, 0.0)

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
