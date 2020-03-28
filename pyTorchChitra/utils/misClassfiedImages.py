import torch
import matplotlib.pyplot as plt
import numpy as np

# As of now I don't know how to initialize a dynamic array
my_misclassified_images = torch.rand(25,3,32,32) * 0
ground_truth = torch.rand(25,1)*0
classified_lie = torch.rand(25,1)*0
num_false_images = 0

def misClassfied(model, device, test_loader, req_num_images=25):
  ''' This function will return mis-classfied images. This will be useful in analyzing the
      network especially in choosing the augmentation stratergy. 

  Args:
      model           : model of the network
      device          : CUDA or CPU
      test_loader     : The image loader. Although the name says test_loader, it can be train loader
                        also. Just pass the right loader during function call.
      req_num_images  : Required number of images. 

  Returns:
      ground_truth           : Ground truth label of misclassified images
      classified_lie          : The label which model classified (mis-classified) 
      my_misclassified_images : The array of images, which are misclassified by model
  '''
                 

  global num_false_images
  global ground_truth
  global classified_lie

  # Put the model in to evaluation mode. This would set the batch-normalization and dropouts (if used)
  # to correct values.   
  model.eval()

  with torch.no_grad():

    # Iterate over test_loader, test loader has some 'X' batch size. It is not guaranteed that model
    # will produce 'req_num_images' in one shot. Hence, iterate over the test_loader
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      
      # Get the model ouput for current batch  
      output = model(data)
      pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

      # Get the false indices
      false_picker = torch.flatten(pred)-target
      index = 0

      # In the current batch of images if the image is mis-classified
      for val in false_picker:
        if (val != 0 and num_false_images < 25):
          my_misclassified_images[num_false_images] = data[index, :, :, :]
          ground_truth[num_false_images] = target[index]
          classified_lie[num_false_images] = pred[index]
          num_false_images = num_false_images + 1

        index = index + 1

        # If we have 'req_num_images' then quit the loop and exit      
        if (num_false_images >= 25):
          break
  
  return(ground_truth, classified_lie, my_misclassified_images)


def displayMisClassfiedImages(ground_truth, mean, std, classified_lie, my_misclassified_images):
  ''' This will use plot the 25 mis-classified images using pyplot and subplots

  Args:
      ground_truth           : Ground truth label of misclassified images
      classified_lie          : The label which model classified (mis-classified) 
      my_misclassified_images : The array of images, which are misclassified by model
      mean                   : Dataset mean
      std                    : Dataset standard deviation

  Returns:
      None
      Displays images
  '''

  # Grid size
  num_img_rows = 5
  num_img_cols = 5

  fig = plt.figure()
  fig.set_figheight(10)
  fig.set_figwidth(10)

  # Class labels
  classes = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

  for i in range(25):
    plt.subplot(5,5,i+1)
    plt.tight_layout()
    image = np.transpose(my_misclassified_images[i,:,:,:], (1, 2, 0))

    # Multiply by its variance and add its mean which is substracted during torch.Normalize()
    image[:,:,0] = (image[:,:,0] * std[0]) + mean[0]
    image[:,:,1] = (image[:,:,1] * std[1]) + mean[1]
    image[:,:,2] = (image[:,:,2] * std[2]) + mean[2]

    plt.imshow(image, interpolation='none')
    plt.title("GT:{}, PRED:{}".format(classes[ground_truth[i]], 
                                      classes[classified_lie[i]]))
    plt.xticks([])
    plt.yticks([])