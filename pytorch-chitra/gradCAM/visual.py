import cv2
import torch

import matplotlib.pyplot as plt

from gradCAM.gradcam import GradCAM
from gradCAM.gradcam_pp import GradCAMPP
from utils.utils import to_numpy, unnormalize


def visualize_cam(mask, img, alpha=1.0):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.

    Args:
        mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]
    Returns:

        heatmap (torch.tensor): heatmap img shape of (3, H, W)
        result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
    """

    heatmap = (255 * mask.squeeze()).type(torch.uint8).cpu().numpy()
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b]) * alpha

    result = heatmap + img.cpu()
    result = result.div(result.max()).squeeze()

    return heatmap, result


class GradCAMView:

    def __init__(self, model, layers, device, mean, std):
        """Instantiate GradCAM and GradCAM++.

        Args:
            model: Trained model.
            layers: List of layers to show GradCAM on.
            device (str or torch.device): GPU or CPU.
            mean: Mean of the dataset.
            std: Standard Deviation of the dataset.
        """
        self.model = model
        self.layers = layers
        self.device = device
        self.mean = mean
        self.std = std

        self._gradcam()
        self._gradcam_pp()

        print('Mode set to GradCAM.')
        self.grad = self.gradcam.copy()

        self.views = []

    def _gradcam(self):
        """ Initialize GradCAM instance. """
        self.gradcam = {}
        for layer in self.layers:
            self.gradcam[layer] = GradCAM(self.model, layer)
    
    def _gradcam_pp(self):
        """ Initialize GradCAM++ instance. """
        self.gradcam_pp = {}
        for layer in self.layers:
            self.gradcam_pp[layer] = GradCAMPP(self.model, layer)
    
    def switch_mode(self):
        """ Switch between GradCAM and GradCAM++. """
        if self.grad == self.gradcam:
            print('Mode switched to GradCAM++.')
            self.grad = self.gradcam_pp.copy()
        else:
            print('Mode switched to GradCAM.')
            self.grad = self.gradcam.copy()
    
    def _cam_image(self, norm_image, class_idx=None):
        """Get CAM for an image.

        Args:
            norm_image: Normalized image. Should be of type
                torch.Tensor
        
        Returns:
            Dictionary containing unnormalized image, heatmap and CAM result.
        """
        image = unnormalize(norm_image, self.mean, self.std)  # Unnormalized image
        norm_image_cuda = norm_image.clone().unsqueeze_(0).to(self.device)
        heatmap, result = {}, {}
        for layer, gc in self.gradcam.items():
            mask, _ = gc(norm_image_cuda, class_idx=class_idx)
            cam_heatmap, cam_result = visualize_cam(
                mask,
                image.clone().unsqueeze_(0).to(self.device)
            )
            heatmap[layer], result[layer] = to_numpy(cam_heatmap), to_numpy(cam_result)
        return {
            'image': to_numpy(image),
            'heatmap': heatmap,
            'result': result
        }
    
    def cam(self, norm_img_class_list):
        """Get CAM for a list of images.

        Args:
            norm_img_class_list: List of dictionaries or list of images.
                If dict, each dict contains keys 'image' and 'class'
                having values 'normalized_image' and 'class_idx' respectively.
                class_idx is optional. If class_idx is not given then the
                model prediction will be used and the parameter should just be
                a list of images. Each image should be of type torch.Tensor
        """
        for norm_image_class in norm_img_class_list:
            class_idx = None
            norm_image = norm_image_class
            if type(norm_image_class) == dict:
                class_idx, norm_image = norm_image_class['class'], norm_image_class['image']
            self.views.append(self._cam_image(norm_image, class_idx=class_idx))
    
    def __call__(self, norm_img_class_list):
        """Get GradCAM for a list of images.

        Args:
            norm_img_class_list: List of dictionaries or list of images.
                If dict, each dict contains keys 'image' and 'class'
                having values 'normalized_image' and 'class_idx' respectively.
                class_idx is optional. If class_idx is not given then the
                model prediction will be used and the parameter should just be
                a list of images. Each image should be of type torch.Tensor
        """
        self.cam(norm_img_class_list)
        return self.views


def plot_gradcam(views, layers, ground_truth, classified_lie, plot_path):
    """Plot heatmap and CAM result.

      Args:
          plot_path: Path to save the plot.
          layers: List of layers.
          ground_truth: Ground truth of the image
          classified_lie: The wrong prediction by model
          view: List of dictionaries containing image, heatmap and result.
    """
    classes = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # Grid size
    num_rows = 25
    num_cols = 5

    fig, axs = plt.subplots(num_rows,num_cols, figsize=(20, 100), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .5, wspace=.001)

    # Make it single tensor so that we can iterate over a for loop
    axs = axs.ravel()

    for row in range(25):
        image_idx = row*5
        img_n_cam = views[row]
        axs[image_idx+0].imshow(img_n_cam['image'])
        axs[image_idx+0].axis('off')
        axs[image_idx+0].set_title("GT:{}, PRED:{}".format(classes[ground_truth[row]], 
                                      classes[classified_lie[row]]))
        axs[image_idx+1].imshow(img_n_cam['result']['layer1'])
        axs[image_idx+1].axis('off')
        axs[image_idx+2].imshow(img_n_cam['result']['layer2'])
        axs[image_idx+2].axis('off')
        axs[image_idx+3].imshow(img_n_cam['result']['layer3'])
        axs[image_idx+3].axis('off')
        axs[image_idx+4].imshow(img_n_cam['result']['layer4'])
        axs[image_idx+4].axis('off')
  
    plt.savefig('./gdrive/My Drive/EVA_Library/Cam_image.png')
    plt.show()
    plt.clf()


'''
def plot_view(layers, view, ground_truth, classified_lie, idx, fig, row_num, ncols, metric):
    """Plot a CAM view.

    Args:
        layers: List of layers
        view: Dictionary containing image, heatmap and result.
        ground_truth: Ground truth of the image
        classified_lie: The wrong prediction by model
        idx : Index for the image (GT/classified lie)
        fig: Matplotlib figure instance.
        row_num: Row number of the subplot.
        ncols: Total number of columns in the subplot.
        metric: Can be one of ['heatmap', 'result'].
    """

    classes = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    sub = fig.add_subplot(row_num, ncols, 1)
    sub.axis('off')
    plt.imshow(view['image'])
    sub.set_title("GT:{}, PRED:{}".format(classes[ground_truth[idx]], 
                                      classes[classified_lie[idx]]))
    for idx, layer in enumerate(layers):
        sub = fig.add_subplot(row_num, ncols, idx + 2)
        sub.axis('off')
        plt.imshow(view[metric][layer])
        sub.set_title(layer)

    
def plot_gradcam(views, layers, ground_truth, classified_lie, plot_path):
    """Plot heatmap and CAM result.

    Args:
        plot_path: Path to save the plot.
        layers: List of layers.
        ground_truth: Ground truth of the image
        classified_lie: The wrong prediction by model
        view: List of dictionaries containing image, heatmap and result.
    """

    for idx, view in enumerate(views):
        # Initialize plot
        fig = plt.figure(figsize=(10, 10))

        # Plot view
        plot_view(layers, view, ground_truth, classified_lie, idx, fig, 1, len(layers) + 1, 'heatmap')
        plot_view(layers, view, ground_truth, classified_lie, idx, fig, 2, len(layers) + 1, 'result')
        
        # Set spacing and display
        fig.tight_layout()
        plt.show()

        # Save image
        # fig.savefig(f'{plot_path}_{idx + 1}.png', bbox_inches='tight')

        # Clear cache
        plt.clf()
'''