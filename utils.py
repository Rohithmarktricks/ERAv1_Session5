import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_device():
    '''Utility function to get the device for training and inferencing of the model.'''
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    return device

def get_transforms(train=True):
    '''Returns the required transforms for train and test data preprocessing/augmentation.'''
    if train:
        transform = transforms.Compose([
            transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
            transforms.Resize((28, 28)),
            transforms.RandomRotation((-15., 15.), fill=0),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
    
    return transform        



def get_mnist_data(train=True, transforms=None):
    '''Retuns the train and test datasets by applying the corresponding transforms'''
    if train:
        dataset = datasets.MNIST('../data',
                                    train=True,
                                    download=True,
                                    transform=transforms)
        
    else:
        dataset = datasets.MNIST('../data',
                                   train=False,
                                   download=True,
                                   transform=transforms)

    return dataset

def get_hyperparams(batch_size):
    '''Returns the required hyperparameters'''
    kwargs = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': 2,
        'pin_memory': True
    }

    return kwargs

def get_dataloader(dataset, hyper_params):
    '''Returns the dataloader for train/test datasets.'''
    return torch.utils.data.DataLoader(dataset, **hyper_params)


def plot_sample(dataloader):
    '''Plots a sample of 12 images along with labels from a given dataloader: train/test.'''
    batch_data, batch_label = next(iter(dataloader))

    fig = plt.figure()

    for i in range(12):
        plt.subplot(3,4,i+1)
        plt.tight_layout()
        plt.imshow(batch_data[i].squeeze(0), cmap='gray')
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])


class Trainer:
    '''Base class for the trainer object.
    Basically, it's a wrapper that takes in the model(to be trained).
    This wrapper can be used to 
    train the model, test, trace the metrics and plot the metrics.
    '''
    def __init__(self, model):
        '''Initializes the trainer object with the given model, and requried metrics to trace the loss/acc.'''
        self.model = model
        self.train_losses = []
        self.train_acc = []
        self.test_losses = []
        self.test_acc = []

    @staticmethod
    def get_correct_pred_count(pred, target):
        '''Retuns the count of the correct predictions'''
        return pred.argmax(dim=1).eq(target).sum().item()

    def train(self, device, train_loader, criterion, optimizer, epoch):
        '''Training utility function'''

        self.model.train()
        pbar = tqdm(train_loader)

        train_loss = 0
        correct = 0
        processed = 0

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            # Predict
            pred = self.model(data)

            # Calculate loss
            loss = criterion(pred, target)
            train_loss+=loss.item()


            # Backpropagation
            loss.backward()
            optimizer.step()
            
            correct += self.get_correct_pred_count(pred, target)
            processed += len(data)

            pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

        self.train_acc.append(100*correct/processed)
        self.train_losses.append(train_loss/len(train_loader))
    
    def test(self,device, test_loader, criterion):
        '''Testing utility function'''
        self.model.eval()

        test_loss = 0
        correct = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)

                output = self.model(data)
                # test_loss += criterion(output, target, reduction='sum').item()
                test_loss += criterion(output, target)
                pred = output.argmax(dim=1, keepdim=True) # get the index of max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

                # correct += self.get_correct_pred_count(output, target)
        
        test_loss /= len(test_loader.dataset)
        self.test_losses.append(test_loss)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        
        self.test_acc.append(100. * correct / len(test_loader.dataset))


    def get_stats(self):
        '''Returns the stats traced during training and testing.'''
        return self.train_losses, self.train_acc, self.test_losses, self.test_acc
    

    def plot_metrics(self):
        '''Plots the trainer metrics: train_loss, train_acc, test_loss, test_acc'''
        test_losses = list(map(lambda x: x.cpu().item(), self.test_losses))
        fig, axs = plt.subplots(2,2,figsize=(15,10))
        axs[0, 0].plot(self.train_losses)
        axs[0, 0].set_title("Training Loss")
        axs[1, 0].plot(self.train_acc)
        axs[1, 0].set_title("Training Accuracy")
        axs[0, 1].plot(test_losses)
        axs[0, 1].set_title("Test Loss")
        axs[1, 1].plot(self.test_acc)
        axs[1, 1].set_title("Test Accuracy")