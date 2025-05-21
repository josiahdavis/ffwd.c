# ffwd.c

Train your model in PyTorch. 

Make inference in pure C.

Simple feedforward network.

<img width="503" alt="image" src="https://github.com/user-attachments/assets/b7816dcc-1cc8-48a4-a4d4-49693e8ac9d0" />

## Features

* Train a deep learning model using PyTorch and save weights.
* Load pretrained model and perform inference in pure C.
* Supports Linear layers (weight & bias) and relu activation function.

## Usage

Step 0: setup environment

```
conda create -yn ffwd-c python=3.12
conda activate ffwd-c
pip install torch numpy pandas
```

Step 1: download california housing dataset ([source](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html))

```
wget -O /tmp/cal_housing.tgz https://www.dcc.fc.up.pt/\~ltorgo/Regression/cal_housing.tgz
tar -xzf /tmp/cal_housing.tgz -C /tmp/
```

Step 1: train the neural network in PyTorch and save weights:

```
python train.py
```

Step 2: run inference in pure C

```
gcc ffwd.c -o ffwd
./ffwd
```