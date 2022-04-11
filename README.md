# DeepDeformationMapRegistration
Deep learning powered image to image registration, based on VoxelMorph

# Usage
The core functionality reside in DeepDeformationMapRegistration package, e.g., the augmentation layers, metrics, etc.

Use the "MultiTrain" scripts to launch the trainings, providing the neccesary parameters. Those in the COMET folder accepts a .ini configuration file (see COMET/train_config_files for example configurations).

Use Evaluate_network to test the trained models. On the Brain folder, use "Evaluate_network__test_fixed.py" instead.

# Warning
The code in this repo is in need of a thorough clean and refactoring. So expect the unexpected.
