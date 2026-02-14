cd ..
git clone git@github.com:565353780/base-gs-trainer.git
git clone --depth 1 https://github.com/facebookresearch/pytorch3d.git
git clone --depth 1 --recursive https://github.com/NVlabs/tiny-cuda-nn.git

cd base-gs-trainer
./dev_setup.sh

cd ../pytorch3d
python setup.py install

cd ../tiny-cuda-nn/bindings/torch
python setup.py install

pip install -e .
