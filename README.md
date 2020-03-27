# pytorch-chitra
This library is built on top of Pytorch. Currently it is being developed for for Cifar-10 dataset. More datasets will be added as the work progresses.

# Features
* Supports Image Augmentations
  * [torchvision.transforms](https://pytorch.org/docs/stable/torchvision/transforms.html)
 * [Albumentations](https://github.com/albumentations-team/albumentations)
Note: Not all the features are supported on the fly. Some modifications may be necessary

* Supported Networks
 * Supports classical CNN
 * Supports ResNet18

* Supports LRFinder

* Supports [Reduce LR On Plateu](https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau)

# Installations 

```python
 pip install -i https://test.pypi.org/simple/ pytorch-chitra-aravinddcsadguru
```

## What is the pytorch-chitra
* Short Answer
  I was not finding _unique_ name in PyPI

* Long answer
  In kannada **Chitra**  means **Image**. As this library belongs to classification of images and built over pytorch, i thought of naming it as pytorch-chitra!
