#!/bin/bash

# set up dependencies manually due to mixture of python2 and python3

# download iacorpus dataset
if [ ! -d data/iac_v1.1 ]; then
    # get dataset and code
    wget http://nldslab.soe.ucsc.edu/iac/iac_v1.1.zip && unzip iac_v1.1.zip -d data && rm iac_v1.1.zip
fi

# load dataset into python-loadable
if [ ! -d data/iac ]; then
    # pickle dump concise dataset
    cp load_iac.py data/iac_v1.1/code
    cd data/iac_v1.1/code && python2 load_iac.py

    # restructure data folder
    cd ../../ && mkdir iac
    mv iac_v1.1/*pkl iac
    cd ..
fi

# perform preprocessing
python preprocess.py -train_file data/iac/train.pkl -valid_file data/iac/val.pkl -save_dir data/iac