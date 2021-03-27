# Rough description of dataset generation

 1. Create Metadata
    - Constructs patient lists with filepaths
    - Reads terminal ileum locations and inflamation severity from csv
 2. Load patient images
    - By default, only loads Axial T2 images
 3. Create Preprocessor
    - Takes size of desired output, which data is resampled to later.
 4. Apply Preprocessor
    - For patient specific localisation, ileum_crop=True, region_grow_crop=False, statistical_region_crop=False
      + Crop image to hardcoded physical size around given ileum location
    - For population-based localisation, ileum_crop=False, region_grow_crop=True, statistical_region_crop=True
      + Use region-growing from seed points in the centre of the axial scan to crop it to the patient
        - Also used to find relative positions of ileum's for manual average calculation
      + Use hard-coded ileum proportional locations to crop image to rough terminal ileum
    - Resampling to reference size is shared
      + Create empty image with same physical size, but reference voxel dimensions and altered spacing.
        - (reference volume)
      + Compute physical center of reference volume
