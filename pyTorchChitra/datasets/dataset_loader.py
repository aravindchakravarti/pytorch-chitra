from torchvision import datasets, transforms
from pyTorchChitra.utils.utils import isCudaAvailable
from albumentations import Compose, RandomCrop, Normalize, HorizontalFlip, Resize, OneOf, Cutout, VerticalFlip, Rotate
from albumentations import ShiftScaleRotate, PadIfNeeded, RandomCrop
from albumentations.pytorch import ToTensor
from PIL import ImageFile, Image
import numpy as np
import torch
import zipfile
import requests
import os
from io import StringIO, BytesIO
from .tiny_imagenet_dataset_format import format_tiny_imgnet_data

'''
class MyTransform(object):
    def __call__(self, img):
        aug = strong_aug(p=1.0)
        return Image.fromarray(augment(aug, np.array(img)))

def augment(aug, image):
    return aug(image=image)['image']

def strong_aug(p=1.0):
        return Compose([
            PadIfNeeded(min_height=36, min_width=36, p=1),
            RandomCrop(height=32, width=32, p=1),
            HorizontalFlip(p=0.5),
            Cutout(num_holes=1, max_h_size=8, max_w_size=8, fill_value=127, p=0.5),
        ], p=1.0)

class dataset_loader(MyTransform):

    cuda = isCudaAvailable()
    dataloader_args = dict(shuffle=True, batch_size=512, num_workers=4, pin_memory=True) \
        if cuda else dict(shuffle=True, batch_size=64)

    def __init__(self, dataset, aug_strategy, batch_size=512):

        self.dataset = dataset
        if (self.dataset == 'tiny_imagenet'):
            self.data_dir = '/content/tiny-imagenet-200'
            self._dataset_downloader('http://cs231n.stanford.edu/tiny-imagenet-200.zip')

        self.batch_size = batch_size

        self.aug_strategy = aug_strategy
        if (self.aug_strategy == 'Albumentations'):
            self.train_transforms = self._apply_albumentations()
        else:
            print (self.aug_strategy + ' is not defined. Please define before proceeding')

        if (self.dataset == 'tiny_imagenet'):
            self.train = datasets.ImageFolder(os.path.join(self.data_dir, 'train'), self.train_transforms)
        
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train, **dataset_loader.dataloader_args)


    @staticmethod
    def _apply_albumentations():
        train_transforms = transforms.Compose([
                                    MyTransform(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                    ])
        return train_transforms

    @staticmethod
    def _dataset_downloader(url):
        if (os.path.isdir("tiny-imagenet-200.zip")):
            print("Files are already downloaded...")
            return
        
        r = requests.get(url, stream = True)
        print ('Downloading ' + url)
        zip_ref = zipfile.ZipFile(BytesIO(r.content))
        zip_ref.extractall('./')
        zip_ref.close()
'''

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
    def strong_aug(p=1.0):
        return Compose([
            PadIfNeeded(min_height=36, min_width=36, p=1),
            RandomCrop(height=32, width=32, p=1),
            HorizontalFlip(p=0.5),
            #Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
            Cutout(num_holes=1, max_h_size=8, max_w_size=8, fill_value=127, p=0.1),
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


def tiny_imagenet_with_albumentations(batch_size = 64, number_of_workers = 4, prob_transform = 1.0):
    '''Downloads the tiny imagenet dataset from the internet if not already available and then applies transforms

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
    def strong_aug(p=1.0):
        return Compose([
            PadIfNeeded(min_height=36, min_width=36, p=1),
            RandomCrop(height=32, width=32, p=1),
            HorizontalFlip(p=0.5),
            Cutout(num_holes=1, max_h_size=8, max_w_size=8, fill_value=127, p=0.4),
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
                                            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
                                            ])

    # Test Phase transformations
    test_transforms = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
                                            ])
    
    print ('Now downloading and allocating dataset')

    # Do we have CUDA support?
    cuda = isCudaAvailable()
    print ("Cuda Available?", cuda)

    # Downloading the dataset if not done already, else fetching from the cache location
    def dataset_downloader(url):
        if (os.path.isdir("tiny-imagenet-200")):
            print("Files are already downloaded...")
            return
            
        r = requests.get(url, stream = True)
        print ('Downloading ' + url)
        zip_ref = zipfile.ZipFile(BytesIO(r.content))
        zip_ref.extractall('./')
        zip_ref.close()

    dataset_downloader('http://cs231n.stanford.edu/tiny-imagenet-200.zip')    

    format_tiny_imgnet_data()

    train = datasets.ImageFolder(os.path.join('./tiny-imagenet-200', 'train'), train_transforms)
    test = datasets.ImageFolder(os.path.join('./tiny-imagenet-200', 'val'), train_transforms)
    
    dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=number_of_workers, pin_memory=True) \
    if cuda else dict(shuffle=True, batch_size=64)

    print ('Now allocating Dataloaders')

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(dataset=train, **dataloader_args)
    test_loader = torch.utils.data.DataLoader(dataset=test, **dataloader_args)

    return train_loader, test_loader

