import utils
import engine
import model_builder
import torch
import data_setup

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

best_model_path = "models/resnet50_imgmet_checkpoint.pth" 
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
best_model = model_builder.resnet50_sent2().to(device)

# Load the saved best model state_dict()
best_model.load_state_dict(torch.load(best_model_path))
data_path = 'data_dir/eurosat/2750'
batch_size = 64
image_net_train_dataloader, image_net_val_dataloader,  image_net_test_dataloader = data_setup.create_dataloaders(
    dataset_path = data_path,
    batch_size = batch_size,
    pre_train_type ='imagenet')


sentinel_train_dataloader, sentinel_val_dataloader, sentinel_test_dataloader = data_setup.create_dataloaders(
    dataset_path = data_path,
    batch_size = batch_size,
    pre_train_type = 'sentinel')

utils.set_seeds(seed=42) 
test_loss, test_acc = engine.test_step(model= best_model,
                dataloader = sentinel_test_dataloader, 
                loss_fn=torch.nn.CrossEntropyLoss(),
                device=device) 
print (f"test_loss: {test_loss:.4f} | "
       f"test_acc: {test_acc:.4f}")