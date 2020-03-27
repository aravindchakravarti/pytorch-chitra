from torchvision import datasets, transforms
from utils.utils import isCudaAvailable
from albumentations import Compose, RandomCrop, Normalize, HorizontalFlip, Resize, OneOf, Cutout, VerticalFlip, Rotate
from albumentations import ShiftScaleRotate
from albumentations.pytorch import ToTensor
from PIL import ImageFile, Image
import numpy as np
import torch
# Get the latest version of Ablumentation library, as standard version is not having good support for cutoout

def getCifar10Data(batch_size = 64, number_of_workers = 4):
  '''Downloads the CIFAR10 dataset from the internet if not already available and then applies transforms

    Args:
      batch_size        : Number of images sent in one iteration of a epoch
      number_of_workers : The workers who are shifting the data from CPU to GPU

    Returns:
      train_loader      : Dataset loader from Torch. The train functions can iterate on
      test_loader       : Dataset loader from Torch. The test functions can iterate on
  '''

  # Train phase transforms
  train_transforms = transforms.Compose([
                                       transforms.RandomAffine(degrees=10, shear = 10),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                       ])

  # Test Phase transformations
  test_transforms = transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                       ])

  print ('Now downloading and allocating dataset')

  # Do we have CUDA drivers for us?
  cuda = isCudaAvailable()
  print ("Cuda Available?", cuda)

  # Downloading the dataset if not done already, else fetching from the cache location
  train = datasets.CIFAR10(root = './data', train=True, download=True, transform=train_transforms)
  test = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transforms)

  dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=number_of_workers, pin_memory=True) \
  if cuda else dict(shuffle=True, batch_size=64)

  print ('Now allocating Dataloaders')

  # Dataloaders
  train_loader = torch.utils.data.DataLoader(dataset=train, **dataloader_args)
  test_loader = torch.utils.data.DataLoader(dataset=test, **dataloader_args)

  return train_loader, test_loader
  


def cifar10WithAlbumentations(batch_size = 64, number_of_workers = 4, prob_transform = 1.0):
  '''Downloads the CIFAR10 dataset from the internet if not already available and then applies transforms

      Args:
        batch_size        : Number of images sent in one iteration of a epoch
        number_of_workers : The workers who are shifting the data from CPU to GPU
        prob_transform     : Probability of applying all the transforms
    
      Returns: 
        train_loader      : Dataset loader from Torch. The train functions can iterate on
        test_loader       : Dataset loader from Torch. The test functions can iterate on
  '''

  # Torch dataloader doesn't support albumentations out of the box. we need a wrapper function which can use __call__ 
  # method.  
  def strong_aug(p=0.5):
    return Compose([
      HorizontalFlip(),
      VerticalFlip(),
      ShiftScaleRotate(rotate_limit=30), 
      Cutout(num_holes=4, max_h_size=6, max_w_size=6),
      Rotate(limit=30) 
      ], p=p)

  def augment(aug, image):
    return aug(image=image)['image']

  class MyTransform(object):
    def __call__(self, img):
      aug = strong_aug(p=prob_transform)
      return Image.fromarray(augment(aug, np.array(img)))

  # Albumentation supports much more transforms than normal torch transports. Hence use that
  # To Do: Move everything to Albumentation
  train_transforms = transforms.Compose([
                                        MyTransform(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                        ])

    # Test Phase transformations
  test_transforms = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                        ])
    
  print ('Now downloading and allocating dataset')

  # Do we have CUDA drivers for us?
  cuda = isCudaAvailable()
  print ("Cuda Available?", cuda)

  # Downloading the dataset if not done already, else fetching from the cache location
  train = datasets.CIFAR10(root = './data', train=True, download=True, transform=train_transforms)
  test = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transforms)

  dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=number_of_workers, pin_memory=True) \
  if cuda else dict(shuffle=True, batch_size=64)

  print ('Now allocating Dataloaders')

  # Dataloaders
  train_loader = torch.utils.data.DataLoader(dataset=train, **dataloader_args)
  test_loader = torch.utils.data.DataLoader(dataset=test, **dataloader_args)

  return train_loader, test_loader


def imshow(img):
  img = img / 2 + 0.5     # unnormalize
  npimg = img.numpy()
  plt.imshow(np.transpose(npimg, (1, 2, 0)))


  # get some random training images
  dataiter = iter(train_loader)
  images, labels = dataiter.next()

  # show images <Enable below instructions if data visualization required>
  imshow(torchvision.utils.make_grid(images))
  # print labels
  print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
