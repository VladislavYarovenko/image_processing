# DL-Inference
This project is based on this [paper](https://arxiv.org/pdf/1805.04252.pdf). The original code can be found [here](https://zenodo.org/record/1242583#.WvAmFXUvz80).
## Requirements 
⋅⋅* Python 2 - 3.7
⋅⋅* Tensorflow 1.3 - 1.15
⋅⋅* Jupyter Notebook
⋅⋅* MySQL
⋅⋅* sklearn, numpy
⋅⋅* Pyro4
⋅⋅* MatPlotLib
## Setup
Adding image dataset ILSVRC 2012 (50K images) to the project

In the value folder (/images/val), images (/images/val/images/) folder should be created.
Inside this folder, images sould be placed in such manner: images/val/images/ILSVRC2012_val_00000001.JPEG.
Image dataset (ImageNet LSVRC 2012) can be downloaded [here](https://academictorrents.com/details/5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5) (Click "I accept the terms" and download the torrent file).

Adding checkpoints for the pre-trained DNNs

In the project folder (/), model_data (/model_data/) folder should be created.
Inside /model_data/tensorflow/checkpoints/ folder, three folders should be created: /inception_v4/, /resnet_v1_152/, /mobilenet_v1_1.0_224/.
Inside these folders, checkpoints should be placed respectively like this: model_data/tensorflow/checkpoints/inception_v4/inception_v4.cpkt (mobilenet will have three files (3 x mobilenet_v1_1.0_224.cpkt.*)).
Checkpoints can be downloaded from [here](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models).
## Running the Project
First, the Database Tutorial.ipynb file should be run, and then the CSV Tutorial.ipynb file. After the two CSV files are generated (they are provided here), Cross validating the premodel.ipynb file should be run.
