import os

import torch

from torchvision import transforms

import data_setup, engine, model_builder, utils

# Setup directories
#train_dir = "data/pizza_steak_sushi/train"
#test_dir = "data/pizza_steak_sushi/test"
#r'C:\Users\bccpe\EuroSAT_ResNet50\EuroSat-with-ResNet50-1\data_dir\eurosat\2750'
data_path = 'data_dir/eurosat/2750'

# Setup target device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


batch_size = 64
num_workers = 2
#os.cpu_count() 
# Create transforms

image_net_train_dataloader, image_net_val_dataloader,  image_net_test_dataloader = data_setup.create_dataloaders(
    dataset_path = data_path,
    batch_size = batch_size,
    pre_train_type ='imagenet')


sentinel_train_dataloader, sentinel_val_dataloader, sentinel_test_dataloader = data_setup.create_dataloaders(
    dataset_path = data_path,
    batch_size = batch_size,
    pre_train_type = 'sentinel'
)

resnet50_imgnet = model_builder.resnet50_imgnet().to(device)
resnet50_sent2 = model_builder.resnet50_sent2().to(device)



# Setup hyperparameters
num_epochs = 5
learning_rate = 0.001



experiment_number = 0
models = [resnet50_imgnet, resnet50_sent2]
utils.set_seeds(seed=42) 

for model in models:
    experiment_number += 1
    print(f"[INFO] Experiment number: {experiment_number}")
    print(f"[INFO] Number of epochs: {num_epochs}")

    if model == resnet50_imgnet:
        print(f"[INFO] Model: {'ResNet50 pretrained with Imagenet'}")
        engine.train(model=model,
                train_dataloader = image_net_train_dataloader,
                test_dataloader = image_net_val_dataloader,
                loss_fn = torch.nn.CrossEntropyLoss(),
                optimizer = torch.optim.SGD(resnet50_imgnet.parameters(), lr=learning_rate),
                epochs = num_epochs,
                device = device,
                writer = utils.create_writer(experiment_name='Exp1 - ResNet50 pretrained with Imagenet',
                                        model_name='resnet50_imgmet',
                                        extra=f"{num_epochs}_epochs"))
        save_filepath = "resnet50_imgmet_checkpoint.pth"
        utils.save_model(model=model,
                       target_dir="models",
                       model_name=save_filepath)
        print("-"*50 + "\n")
        
    else:
        print(f"[INFO] Model: {'ResNet50 pretrained with Sentinel2'}")
        engine.train(model=model,
                train_dataloader = sentinel_train_dataloader,
                test_dataloader= sentinel_val_dataloader,
                loss_fn = torch.nn.CrossEntropyLoss(),
                optimizer = torch.optim.SGD(resnet50_sent2.parameters(), lr=learning_rate),
                epochs = num_epochs,
                device = device,
                writer = utils.create_writer(experiment_name='Exp2 - ResNet50 pretrained with Sentinel2 ',
                                        model_name='resnet50_sent2',
                                        extra=f"{num_epochs}_epochs"))
        save_filepath = "resnet50_sent2_checkpoint.pth"
        utils.save_model(model=model,
                       target_dir="models",
                       model_name=save_filepath)
        print("-"*50 + "\n")
        