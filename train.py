import os

import torch

from torchvision import transforms

import data_setup, engine, model_builder, utils

# Setup directories
#train_dir = "data/pizza_steak_sushi/train"
#test_dir = "data/pizza_steak_sushi/test"
data_path = 'data/eurosat/2750'

# Setup target device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create transforms

image_net_train_dataloader, image_net_val_dataloader,  image_net_test_dataloader = data_setup.create_dataloaders(
    data_dir = data_path,
    batch_size = batch_size,
    num_workers = ''
    pre_train_type ='imagenet')


sentinel_train_dataloader, sentinel_val_dataloader, sentinel_test_dataloader = data_setup.create_dataloaders(
    data_dir = data_path,
    batch_size = batch_size,
    num_workers = ''
    pre_train_type = 'sentinel'
)

resnet50_imgnet = model_builder.resnet50_imgnet().to(device)
resnet50_sent2 = model_builder.resnet50_sent2().to(device)



# Setup hyperparameters
num_epochs = 5
batch_size = 64
hidden_units = 10
learning_rate = 0.001

# Specify criterion and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet50_imgnet.parameters(), lr=lr)

experiment_number = 0
models = [resnet50_imgnet, resnet50_sent2]

for model in models:
    experiment_number += 1
    print(f"[INFO] Experiment number: {experiment_number}")
    print(f"[INFO] Model: {str(model)")
    print(f"[INFO] Number of epochs: {num_epochs}")

    if model == resnet50_imgnet:
        engine.train(model=model,
                train_dataloader = image_net_train_dataloader,
                test_dataloader = image_net_val_dataloader,
                loss_fn = criterion,
                optimizer = optimizer,
                epochs = num_epochs,
                device = device,
                writer = utils.create_writer(experiment_name='exp1',
                                        model_name='restnet',
                                        extra=f"{num_epochs}_epochs"))
        
    else:
        engine.train(model=model,
                train_dataloader = image_net_train_dataloader,
                test_dataloader= image_net_val_dataloader,
                loss_fn = criterion,
                optimizer = optimizer,
                epochs = num_epochs,
                device = device,
                writer = utils.create_writer(experiment_name='exp1',
                                        model_name='restnet',
                                        extra=f"{num_epochs}_epochs")))
        