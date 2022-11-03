<br />
<p align="center">
  
  <h3 align="center">ConvNeXt Backbone for Mask R-CNN</h3>

  <p align="center">
    Easy way to use Mask R-CNN with ConvNeXt backbone.
    <br />
  </p>
</p>


![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)


<!-- ABOUT THE PROJECT -->
## About The Project

This study allows the ConvNeXt architecture for the MaskRCNN model, available in the torchvision library, to be used as a backbone network. It also includes a customized trainer class. The study was also tested in one of the Cell Tracking Challenge datasets. The results of several different backbone network configurations were shared.


<!-- GETTING STARTED -->
## Getting Started

Follow these simple example steps to get a local copy up and to run.


### Installation

1. Clone the repo
```sh
git clone https://github.com/mberkay0/ConvNeXt-MaskRCNN.git
```
2. Check if you have a virtual env 
```sh
virtualenv --version
```
3. If (not Installed) 
```sh
pip install virtualenv
```
4. Now create a virtual env in cd ConvNeXt-MaskRCNN/
```sh
virtualenv venv
```
5. Then download a python modules
```sh
pip install -r requirements.txt
```

<!-- USAGE -->
## Usage

Provide helper functions to simplify writing torchvision pipelines using pre-trained models. Here is how you would do it.

```python
import torch
from torchvision.models.detection import MaskRCNN
from .inference import Config
from .train_utils import convnext_fpn_backbone, Trainer
from .dataset import BuildDataset
from .utils import get_file_dir


train_dataset = BuildDataset(get_file_dir(train_img_path), 
                             get_file_dir(train_mask_path))
train_loader = DataLoader(train_dataset, batch_size=Config.train_bs, 
                          num_workers=4, shuffle=True, pin_memory=True, 
                          drop_last=True, collate_fn=lambda x: tuple(zip(*x)))

backbone = convnext_fpn_backbone(
    Config.backbone,
    Config.trainable_layers
)

model = MaskRCNN(
    backbone, 
    num_classes=Config.num_classes, 
    max_size=Config.max_size,
    min_size=Config.min_size,
)

model.to(Config.device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(
    params, 
    lr=Config.lr, 
    weight_decay=Config.weight_decay
)

lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=Config.step_size,
    gamma=Config.gamma
)
scaler = torch.cuda.amp.GradScaler()
```

Train your MaskRCNN model with the ConvNeXt backbone architecture with the help of the Trainer class in an easy way.

```python
trainer = Trainer(
    optimizer=optimizer,
    max_epochs=Config.epochs,
    device=Config.device,
    scaler=scaler,
    verbose_num=Config.verbose_num,
    split_size=Config.split_size,
    val_bs=Config.val_bs
)

history = trainer.fit(
    model, 
    train_dataloader=train_loader, 
    ckpt_path=Config.save_path + Config.model_name + ".pth"
)
```

<img src="/images/lossplot.png" alt="train-val-loss" width="470" height="390"/>

<!-- RESULTS -->
## Results

Some results from the training using the ConvNeXt backbone network are shown below.

| backbone name | resolution |dice score (%) | number of epoch | 
|:---:|:---:|:---:|:---:|
| ResNet50 | 512x512 | 87.9 | 20 |
| ConvNeXt-B | 512x512 | 92.0 | 20 |
| ConvNeXt-T | 512x512 | 91.6 | 20 | 


<img src="/images/result.png" alt="example-result" width="600" height="500"/>


## References

* [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
* [Official PyTorch implementation of ConvNeXt](https://github.com/facebookresearch/ConvNeXt)
* [Torchvision](https://pytorch.org/vision/stable/index.html)
* [Cell Tracking Challenge](http://celltrackingchallenge.net)
