from preprocess import Preprocessor
from metadata import Metadata
from np_generator import NumpyGenerator

# Reverse-engineer dimensions from desired global average pooling size (assuming three downsampling layers)
pool_size = [10, 10, 10]
input_size = [2 * (2 * (2 * x + 1) + 1) + 1 for x in pool_size]
reference_size = [x + pad for x, pad in zip(input_size, [12, 12, 12])]
k = 4
test_proportion = 0.25
print('input_size', input_size)
print('record_size', reference_size)

# Path setting
data_path = '/vol/bitbucket/mb4617/MRI_Crohns'
label_path = '/vol/bitbucket/mb4617/MRI_Crohns/labels'
record_out_path = '/vol/bitbucket/mb4617/MRI_Crohns/numpy_datasets/ti_imb'
record_suffix = 'all_data'

# Load data
abnormal_cases = list(range(70))
healthy_cases = list(range(100))
metadata = Metadata(data_path, label_path, abnormal_cases, healthy_cases, dataset_tag='')
# metadata = Metadata(data_path, label_path, abnormal_cases, healthy_cases, dataset_tag=' cropped')

print('Loading images...')
for patient in metadata.patients:
    print(f'Loading patient {patient.get_id()}')
    patient.load_image_data(axial=True, coronal=True, axial_postcon=True)
    print()
print()

# Preprocess data
preprocessor = Preprocessor(constant_volume_size=reference_size)
metadata.patients = preprocessor.process(metadata.patients, ileum_crop=True, region_grow_crop=False, statistical_region_crop=False)

# Serialize data into numpy files 
numpy_generator = NumpyGenerator(record_out_path, record_suffix)
numpy_generator.generate_cross_folds(k, metadata.patients)

print('Done')
