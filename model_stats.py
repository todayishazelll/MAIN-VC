import torch
from torchinfo import summary
import yaml
from models.model import MAINVC  # Import MAINVC model

# Load config
with open("./config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Initialize MAINVC model
model = MAINVC(config)

# Move model to CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define input shapes
x = torch.randn(1, 80, 128).to(device)  # Speech feature
x_sf = torch.randn(1, 80, 128).to(device)  # Speaker feature
x_ = torch.randn(1, 80, 128).to(device)  # Speaker reference feature

# Get model summary
summary(model, input_data=[x, x_sf, x_], depth=5)  # Adjust depth for details
