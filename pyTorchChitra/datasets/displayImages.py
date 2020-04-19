import matplotlib.pyplot as plt
import math 
import numpy as np
import os

def displayDataSetSampleImages(image_loader, classes, mean, std, num_of_images):
    '''
    This function plots (or subplots) the images in the form or grid
    the number of rows and number columns are calculated by the input
    parater. If the input cannot form the rectangle, then it will lead
    to problems

    Args:
        image_loader : Torch image loader. It can be train/test loader
        classes      : Class labels if present
        mean         : Dataset mean
        std          : Dataset standard deviation
        num_of_images : Number of images to be displayed

    Output:
        Saves the subplot image in the current directory

    Returns:
        None
    
    '''
    
    dataiter = iter(image_loader)
    images, labels = dataiter.next()

    # pyplot does not accept the shape (3,H,W) it wants (H,W,3)
    images = np.transpose(images, (0, 2, 3, 1))
    
    # Print batch-images size
    print(images.shape)

    # Get the labels
    labels = labels.numpy()

    # Calculate number of rows and columns
    num_rows = int(np.floor(math.sqrt(num_of_images)))
    num_cols = int(num_of_images/num_rows)

    # Setup subplot
    fig, axs = plt.subplots(num_rows,num_cols, figsize=(10, 10), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .5, wspace=.001)

    # Make it single tensor so that we can iterate over a for loop
    axs = axs.ravel()


    for i in range(num_of_images):
        # For each image in the generator
        display_image = images[i,:,:,:]
        # Multiply by its variance and add its mean which is substracted during torch.Normalize()
        display_image[:,:,0] = (display_image[:,:,0] * std[0]) + mean[0]
        display_image[:,:,1] = (display_image[:,:,1] * std[1]) + mean[1]
        display_image[:,:,2] = (display_image[:,:,2] * std[2]) + mean[2]

        # Add image to the subplot
        axs[i].imshow(images[i,:,:,:])

        # Not all dataset comes with labels. For example tiny imagenet has not class label information.
        if (len(classes) != 0):
            axs[i].set_title(classes[labels[i]]) 

        axs[i].axis('off')

    try:
        plt.savefig('./gdrive/My Drive/EVA_Library/Sample_Images.png')
    
    except IOError:
        print('The path does not exist. Try different path in ->displayImages.py<- file')

    plt.clf()