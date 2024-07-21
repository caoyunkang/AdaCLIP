# add dependencies
# python395_cuda113_pytorch1101
# please change dataset root in ./config.py according to your specifications

conda create -n AdaCLIP python=3.9.5 -y
conda activate AdaCLIP
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install tqdm tensorboard setuptools==58.0.4 opencv-python scikit-image scikit-learn matplotlib seaborn ftfy regex numpy==1.26.4
pip install gradio






