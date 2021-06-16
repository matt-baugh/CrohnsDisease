
# CrohnsDisease
Final year Masters project at Imperial College on tackling Crohn's Disease


Follow up project to R. Hollands work:
   - arXiv: https://arxiv.org/abs/1909.00276
   - Paper was presented at MICCAI PRIME 2019, Shenzhen.

Classifying Crohn's disease from MRI volumes, using a 3D ResNet with a soft attention mechanism.
Capable of an average f-1 score of >0.8 using both patient-specific localisation and population-specific localisation.

For description/details of old project, please view the original repository:
[https://github.com/RobbieHolland/CrohnsDisease](https://github.com/RobbieHolland/CrohnsDisease)

## Repo Guide
Brief explanation of important files which are used in this iteration of the project (old files are left unchanged for reference).

### Training
<tt>/run_crohns_pytorch.sh</tt> - Run config specifying training and model parameters (root of execution) for cross-validation experiment.

<tt>/run_crohns_pytorch_all.sh</tt> - Run batch of cross-validation experiments, testing each network configuration (with or without multimodal data, patient-specific localisatino, attention mechanism).

<tt>/run_pytorch.py</tt> - Parses config options and starts training procedure.

<tt>/pytorch/pytorch_train.py</tt> - Constructs and iteratively trains Pytorch network, logging the performance at each step.

<tt>/pytorch/mri_dataset.py</tt> - Loads data saved in dataset by <tt>/preprocessing/np_generator.py</tt>, performs data augmentation.

<tt>/pytorch/resnet.py</tt> - Specification for 3D Resnet, including [soft attention mechanism](https://arxiv.org/abs/1804.05338)

### Preproprocessing pipeline
Files under <tt>/preprocessing/</tt> generate the `.npy` training and testing datasets used in training.

<tt>/preprocessing/metadata.py</tt> - Loads labels and MRI data into memory.

<tt>/preprocessing/preprocess.py</tt> - Extracts region of interest from MRI volumes.

<tt>/preprocessing/np_generator.py</tt> - Generates a series of training and test '.npy' files for cross-fold evaluation.

<tt>/preprocessing/generate_np_datasets.py</tt> - Configures and executes the generation process (i.e. how many cross folds)

### ~Helpful notebooks

They were useful to me, they might be useful to you.

Much of the code is mini-experiments or tests I used when developing the project, so they may not work or serve an obvious purpose now.

Think of them as my scrap paper when solving the problems of this project, so they may not function as the project has iterated over time.

<tt>/preprocessing/multimodal_precossing_test.ipynb</tt> - Notebook going step by step through process in <tt>/preprocessing/generate_np_datasets.py</tt>, so images can be inspected at each stage.

<tt>/pytorch/test_numpy_dataset.ipynb</tt> - Load data into the `MRIDataset` of <tt>/pytorch/mri_dataset.py</tt>, test data augmentation methods.

<tt>/pytorch/view_numpy_dataset.ipynb</tt> - Manually load saved dataset from `.npy` file, visualise examples.

<tt>/pytorch/test_trained_model.py</tt> - Load saved model and relevant dataset for specific experiment, showing results.

<tt>/pytorch/compare_noise_amounts.py</tt> - Load datasets, investigate how noise effects intensity distribution.