import torch
from torchinfo import summary
import yaml
import time
from models.model import MAINVC  # Import MAINVC model

# Start the timer
start_time = time.time()

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

# End the timer and calculate the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time

# Print the running time
print(f"Script execution time: {elapsed_time:.4f} seconds")
