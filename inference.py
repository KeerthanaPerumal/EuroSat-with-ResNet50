import utils
import engine
import model_builder
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

best_model_path = "models/resnet50_sent2_checkpoint.pth"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
best_model = model_builder.resnet50_sent2().to(device)

# Load the saved best model state_dict()
best_model.load_state_dict(torch.load(best_model_path))

utils.set_seeds(seed=42) 
engine.test_step(model= best_model
                dataloader = sentinel_test_dataloader, 
                loss_fn=torch.nn.CrossEntropyLoss(),
                device=device) 