gpu_id=0

# Note: Since we have utilized half-precision (FP16) for training, the training process can occasionally be unstable.
# It is recommended to run the training process multiple times and choose the best model based on performance
# on the validation set as the final model.

# pre-trained on MVtec and colondb
CUDA_VISIBLE_DEVICES=$gpu_id python train.py --save_fig True --training_data mvtec colondb --testing_data visa

# pre-trained on Visa and Clinicdb
CUDA_VISIBLE_DEVICES=$gpu_id python train.py --save_fig True --training_data visa clinicdb --testing_data mvtec

# This model is pre-trained on all available data to create a powerful Zero-Shot Anomaly Detection (ZSAD) model for demonstration purposes.
CUDA_VISIBLE_DEVICES=$gpu_id python train.py --save_fig True \
--training_data \
br35h brain_mri btad clinicdb colondb \
dagm dtd headct isic mpdd mvtec sdd tn3k visa \
--testing_data mvtec

