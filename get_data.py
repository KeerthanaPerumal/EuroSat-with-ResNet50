
from torchvision import datasets
import torch
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

total_data = datasets.EuroSAT(
    root = 'data_dir',
    download = True,
    transform = None,
    target_transform = None
)
