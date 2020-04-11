import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F 
from pyTorchChitra.modelTraining.CNNTrainTest import train, test
import matplotlib.pyplot as plt

def setup_ROP_Optimizer(net, device, optimzer_select='NLLLoss', lr=0.01):
    model =  net.to(device)
    criteria = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = lr, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                    mode = 'min', 
                                                    factor = 0.1,
                                                    verbose = True,
                                                    patience = 2
                                                    )
    return optimizer, scheduler

def setup_OCLR_Optimizer(net, device, total_steps, num_raise, anneal = 'linear', div_factor = 6, max_lr = 0.005):

    model =  net.to(device)
    criteria = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = (max_lr/div_factor), momentum=0.9)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer = optimizer,
                                            max_lr = max_lr,
                                            total_steps = total_steps,
                                            pct_start = num_raise,
                                            anneal_strategy = anneal, 
                                            div_factor = div_factor,
                                            final_div_factor = 1,
                                            )
    return optimizer, scheduler

def runTheModel (model, device, optimizer, train_loader, test_loader, scheduler, epochs = 20, isOCLR = False):
    for epoch in range(epochs):
        print("EPOCH:", epoch)
        plot_train_losses, plot_train_acc = train(model, device, train_loader, optimizer, epoch, scheduler, isOCLR=isOCLR)
        val_loss, plot_test_losses, plot_test_acc = test(model, device, test_loader)
        print('Validation loss to Schedular = '+str(val_loss)+'\n')
        if isOCLR == False:
            scheduler.step(val_loss)

    return(plot_train_losses, plot_train_acc, plot_test_losses, plot_test_acc)

def modelTrainTestHistory(t_l, t_a, v_l, v_a):
    '''
    This function plots the model train       - loss and accuracy
                            model validation  - loss and accuracy

    Args:
        t_l     : Training loss (array)
        t_a     : Training accuracy (array)
        v_l     : Testing loss (array)
        v_a     : Testing accuracy (array)

    Returns:
        None
    
    Outputs:
        pyplot
    '''
  
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(t_l)
    axs[0, 0].set_title("Training Loss")
    axs[0, 0].grid(b=True, which='both')

    axs[1, 0].plot(t_a)
    axs[1, 0].set_title("Training Accuracy")
    axs[1, 0].grid(b=True, which='both')

    axs[0, 1].plot(v_l)
    axs[0, 1].set_title("Test Loss")
    axs[0, 1].grid(b=True, which='both')

    axs[1, 1].plot(v_a)
    axs[1, 1].set_title("Test Accuracy")
    axs[1, 1].grid(b=True, which='both')

    plt.savefig('./gdrive/My Drive/EVA_Library/Model_performance.png')