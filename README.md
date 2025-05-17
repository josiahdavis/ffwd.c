# ffwd.c

Simple feedforward network for inference.

<img width="503" alt="image" src="https://github.com/user-attachments/assets/b7816dcc-1cc8-48a4-a4d4-49693e8ac9d0" />

## Features

* Train a deep learning model using PyTorch and save weights.
* Load weights & data and perform inference in pure C.
* Supports Linear layers (weight & bias) and relu activation function.

## Usage

```
gcc ffwd.c -o ffwd
./ffwd <path_to_saved_data> <path_to_saved_model>
```

Optionally, to train the neural network (in PyTorch):

```
conda create -yn ffwd-c python=3.12
conda activate ffwd-c
pip install torch numpy
python train.py
```

## To do

- [ ] Pick out a dataset
- [ ] PyTorch for training model
- [ ] Load weights in C
- [ ] Load data in C
- [ ] Network definition in C
