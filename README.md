# TSAI - ERA V1 Session 5 Assignment

## Problem Statement
1. Re-look at the code that we worked on in Assignment 4 (the fixed version). 
2. Move the contents of the code to the following files:
    - model.py
    - utils.py
    - S5.ipynb
3. Make the whole code run again. 
4. Upload the code with the 3 files + README.md file (total 4 files) to GitHub. README.md (look at the spelling) must have details about this code and how to read your code (what file does what). Heavy negative scores for not formatting your markdown file into p, H1, H2, list, etc. 
5. Attempt Assignment 5. 

## Approach/Solution
Please refer to ![S5.ipynb](S5.ipynb) for the solution that uses the helper functions from ```utils.py``` and ```model.py```

### Helper Files

### ![utils.py](utils.py)
The ```utils.py``` module contains the following functions and classes.

1. ```get_device()```: This function returns the device (CPU/GPU) required for training/inference
2. ```get_transfroms(train=True)```: This function returns the required train/test transformations along with train/test augmentation.
3. ```get_mnist_data(train=True, transforms=None)```: This function returns the train/test MNIST dataset with corresponding train/test transforms.
4. ```get_hyperparams(batch_size)```: This function returns the required hyperparameters for loading the train/test dataset.
5. ```plot_sample(dataloader)```: This function plots the sample(12) images along with the labels from a given dataloader (train/test)
6. ```Trainer```: Base class that takes the ```model``` and facilitates the training/testing processes. It also traces the loss and accuracy metrics along training/testing.
    1. It contains the following methods:
        - ```__init__```: A initializer that takes the model and also intitilizes required variables to trace the train & test loss/acc.
        - ```train```: A method to train the model and also traces the training loss and accuracy.
        - ```test```: A method to test the model and also traces the testing loss and accuracy.
        - ```get_stats```: Method that retuns all the traced metrics ```train_losses, train_acc, test_losses, test_acc```
        - ```plot_metrics```: Method to plot all the train/test traced metrics. Retuns a plot.

### ![model.py](model.py)
The ```model.py``` module contains the source code for Neural network (along with loss, optimizer and scheduler)
1. ```Net```: A Pytorch NNet of type nn.Module that includes ```nn.Conv2d, nn.Linear, F.relu, F.max_pool2d``` and ```log_softmax```
2. ```get_loss```: A method that returns the ```nn.CrossEntropyLoss()``` object.
3. ```get_optimizer```: A method that returns the optimizer and scheduler required for optimizing the loss function during the backpropagation.


### Metrics
![Loss and Accuracy](/images/metrics.png)