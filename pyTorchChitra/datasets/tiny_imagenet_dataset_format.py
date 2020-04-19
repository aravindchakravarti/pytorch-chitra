import io
import glob
import os
from shutil import move
from os.path import join
from os import listdir, rmdir
import time
import scipy.ndimage as nd
import numpy as np
import random

def format_tiny_imgnet_data():
    target_folder = './tiny-imagenet-200/val/'
    test_folder   = './tiny-imagenet-200/test/'

    #os.mkdir(test_folder)
    val_dict = {}
    with open('./tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
        for line in f.readlines():
            split_line = line.split('\t')
            val_dict[split_line[0]] = split_line[1]
            
    paths = glob.glob('./tiny-imagenet-200/val/images/*')
    for path in paths:
        file = path.split('/')[-1]
        folder = val_dict[file]
        if not os.path.exists(target_folder + str(folder)):
            os.mkdir(target_folder + str(folder))
            os.mkdir(target_folder + str(folder) + '/images')
        if not os.path.exists(test_folder + str(folder)):
            os.mkdir(test_folder + str(folder))
            os.mkdir(test_folder + str(folder) + '/images')
            
            
    for path in paths:
        file = path.split('/')[-1]
        folder = val_dict[file]
        if len(glob.glob(target_folder + str(folder) + '/images/*')) <25:
            dest = target_folder + str(folder) + '/images/' + str(file)
        else:
            dest = test_folder + str(folder) + '/images/' + str(file)
        move(path, dest)
        
    rmdir('./tiny-imagenet-200/val/images')

'''
def format_tiny_imgnet_data():

    target_folder = './tiny-imagenet-200/val/'
    dest_folder = './tiny-imagenet-200/train/'

    val_dict = {}

    with open('./tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
        for line in f.readlines():
            split_line = line.split('\t')
            val_dict[split_line[0]] = split_line[1]

    paths = glob.glob('./tiny-imagenet-200/val/images/*')

    for path in paths:
        files = path.split('/')[-1].split('\\')[-1]
        folder = val_dict[files]
        dest = dest_folder + str(folder) + '/images/' + str(files)
        move(path, dest)

    f.close()

    target_foler = './tiny-imagenet-200/train/'
    train_folder = './tiny-imagenet-200/train_set/'
    test_folder = './tiny-imagenet-200/test_set/'

    os.mkdir(train_folder)
    os.mkdir(test_folder)

    paths = glob.glob('./tiny-imagenet-200/train/*')

    for path in paths:
        folder = path.split('/')[-1].split('\\')[-1]
        source = target_folder + str(folder+'/images/')
        train_dest = train_folder + str(folder+'/')
        test_dest = test_folder + str(folder+'/')
        os.mkdir(train_dest)
        os.mkdir(test_dest)
        images = glob.glob(source+str('*'))

        #Making random
        random.shuffle(images)

        test_imgs = images[:165].copy()
        train_imgs = images[165:].copy()

        #Moving the 30% for validation
        for image in test_imgs:
            files = image.split('/')[-1].split('\\')[-1]
            dest = test_dest+str(files)
            move (image, dest)

        #Moving 70 for training
        for image in train_imgs:
            files = image.split('/')[-1].split('\\')[-1]
            dest = train_dest+str(files)
            move(image, dest)
'''