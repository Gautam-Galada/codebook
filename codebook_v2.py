import torch
import torch.nn as nn
import torch.optim as optim
import os
from PIL import Image
import numpy as np

class Transformer(nn.Module):
    def __init__(self, codebook_size, input_dim, hidden_dim, output_dim):
        super(Transformer, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        self.codebook = nn.Embedding(codebook_size, hidden_dim)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.codebook(torch.clamp(x.argmax(dim=-1), 0, codebook_size-1).unsqueeze(-1))
        x = self.decoder(x.squeeze(-1))
        return x

def train_transformer(transformer, codebook, dataset, num_epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(transformer.parameters()) + [codebook.weight], lr=learning_rate)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for image in dataset:
            optimizer.zero_grad()
            image = image.unsqueeze(0)  # Add a batch dimension
            encoded_image = transformer.encoder(image)
            quantized_image = codebook(torch.clamp(encoded_image.argmax(dim=-1), 0, codebook_size-1).unsqueeze(-1))
            reconstructed_image = transformer.decoder(quantized_image.squeeze(-1))
            loss = criterion(reconstructed_image, image)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

codebook_size = 10
input_dim = 3
hidden_dim = 3
output_dim = 3
num_epochs = 10
learning_rate = 0.001

codebook = nn.Embedding(codebook_size, hidden_dim)
transformer = Transformer(codebook_size, input_dim, hidden_dim, output_dim)

folder_path = 'dataset'
dataset = []
image_size = (64, 64)  # Desired image size

for filename in os.listdir(folder_path):
    image_path = os.path.join(folder_path, filename)
    image = Image.open(image_path).convert("RGB")  # Convert image to RGB if needed
    image = image.resize(image_size)  # Resize image to desired size
    image_tensor = torch.Tensor(np.array(image))
    dataset.append(image_tensor)

dataset = torch.stack(dataset)

train_transformer(transformer, codebook, dataset, num_epochs, learning_rate)
