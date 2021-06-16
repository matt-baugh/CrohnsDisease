# Rough description of dataset generation

General Image Indexing:
 - Terminal Ileum coordinates given as [Coronal, Sagittal, Axial]
 - Indexing with SITK is [Sagittal, Coronal, Axial]
 - Indexing in Numpy is [Axial, Coronal, Sagittal]

 1. Create Metadata
    - Constructs patient lists with filepaths
    - Reads terminal ileum locations and inflamation severity from csv
 2. Load patient images
    - By default, only loads Axial T2 images
 3. Create Preprocessor
    - Takes size of desired output, which data is resampled to later.
 4. Apply Preprocessor
    - For patient specific localisation, ileum_crop=True, region_grow_crop=False, statistical_region_crop=False
      + Region of interest hardcoded physical size around given ileum location
    - For population-based localisation, ileum_crop=False, region_grow_crop=True, statistical_region_crop=True
      + Use region-growing from seed points in the centre of the axial T2 scan to crop it to the patient
        - Also used to find relative positions of ileum's for manual average calculation
      + Region of interest is hardcoded box proportional to axial T2 scan patient bounding box
    - Create reference volume of region of interest, with voxel dimensions set by step 3.
    - Use reference volume to extract region of interest from each scan modality
 5. Generate datasets with NumpyGenerator
    - Split data into folds with equal amounts of each inflammation severity label.
    - Save each train-test split as `.npy` files.
