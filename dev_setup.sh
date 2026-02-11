cd ..
git clone https://github.com/NVlabs/nvdiffrast.git
git clone https://github.com/NVlabs/tiny-cuda-nn.git

pip install torch==2.1.2 torchvision==0.16.2

pip install numpy==1.26.4

cd nvdiffrast
python setup.py install

cd ../tiny-cuda-nn/bindings/torch
python setup.py install

pip install -e .
