# DeepMatting

### Installation
if you want to use pretrained models, then all you need to do is:
```sh
git clone https://github.com/djang000/DeepMatting.git
```

if you also want to train new modes, you will need the portrait images and GT mask images for training and pretrained MnasNet wegihts.

I uploaded my MnasNet weight by trained with ImageNet 2012

### Usage

Following are examples of how the scripts in this repo can be used.

- inspect_model.ipynb

	you can show the evaluation result using trained model.

- train.py

	you just run below commad.

```sh
python train.py
```

