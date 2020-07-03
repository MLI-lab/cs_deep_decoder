# Compressive sensing with un-trained neural networks: Gradient descent finds the smoothest approximation

This repository provides code for reproducing the figures in the  paper:

**``Compressive sensing with un-trained neural networks: Gradient descent finds the smoothest approximation''** by Reinhard Heckel and Mahdi Soltanolkotabi. Contact: [reinhard.heckel@gmail.com](reinhard.heckel@gmail.com)


## Organization

- Figure 1: compressive_sensing_example_convergence.ipynb
- Figure 5: MRI_multicoil_deep_decoder_accelerate.ipynb

## Installation

The code is written in python and relies on pytorch. The following libraries are required: 
- python 3
- pytorch
- numpy
- skimage
- matplotlib
- scikit-image
- jupyter

The libraries can be installed via:
```
conda install jupyter
```

The code to reproduce the MRI experiment uses a few function from the fastMRI repository to load the k-space data, those can be obtained by copying the data and common folders from the repository [https://github.com/facebookresearch/fastMRI](https://github.com/facebookresearch/fastMRI). In particular, download the code from the fastMRI repository, and copy the folder fastMRI/data into the cs_deep_decoder repository.

## Citation
```
@inproceedings{heckel_compressive_2020,
    author    = {Reinhard Heckel and Mahdi Soltanolkotabi},
    title     = {Compressive sensing with un-trained neural networks: {Gradient} descent finds the smoothest approximation},
    booktitle = { {International} {Conference} on {Machine} {Learning} },
    year      = {2020},
}
```

## Licence

All files are provided under the terms of the Apache License, Version 2.0
