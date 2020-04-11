from tqdm import tqdm
import torch.nn.functional as F 
import torch
from pyTorchChitra.utils.global_vars import cyclic_LR_history


train_losses = []
test_losses = []
train_acc = []
test_acc = []

def train(model, device, train_loader, optimizer, epoch, scheduler, isOCLR = False):
  model.train()
  pbar = tqdm(train_loader)

  correct = 0
  processed = 0

  for batch_idx, (data, target) in enumerate (pbar):
    data, target = data.to(device), target.to(device)

    #Don't want history of gradients 
    optimizer.zero_grad()

    y_predict = model(data)

    # Caluclate loss
    loss = F.nll_loss(y_predict, target)
    train_losses.append(loss)

    # Back propogate error
    loss.backward()

    # Take a optimzer step
    optimizer.step()

    # Update pbar-tqdm
    
    pred = y_predict.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)

    if isOCLR == True:
        scheduler.step()
        for param_group in optimizer.param_groups:
            cyclic_LR_history.append(param_group['lr'])

  return(train_losses, train_acc)  


def test (model, device, test_loader):
  model.eval()

  test_loss = 0
  correct = 0

  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)

      output = model(data)

      test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
      pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)

  print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
  
  test_acc.append(100. * correct / len(test_loader.dataset))

  return(test_loss, test_losses, test_acc)