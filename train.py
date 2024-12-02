import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from models.model import ImageCaptioningModel
from utils.preprocess import FlickrDataset
import time
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from utils.vocabulary import build_vocabulary

def train_model(train_loader, model, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for images, captions in train_loader:
        images = images.to(device)
        captions = captions.to(device)
        
        # Forward pass
        outputs = model(images, captions[:, :-1])  # Remove last token from input
        
        # Calculate loss (ignore padding)
        targets = captions[:, 1:]  # Remove first token (usually <START>)
        
        # Ensure shapes match before computing loss
        batch_size, seq_length = targets.shape
        outputs = outputs[:, :seq_length, :]  # Trim outputs to match target sequence length
        
        # Reshape for loss calculation
        outputs = outputs.reshape(-1, outputs.shape[2])
        targets = targets.reshape(-1)
        
        # Compute loss
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def main():
    # Hyperparameters
    embed_size = 256
    hidden_size = 512
    num_layers = 2
    learning_rate = 3e-4
    num_epochs = 10
    batch_size = 64
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load, create and save the vocabulary
    df = pd.read_csv(r"data\captions.csv", delimiter='|')
    captions = df['caption']
    vocab = build_vocabulary(captions, min_word_freq=3, save_path=r'data\vocabulary')
    
    # Create dataset
    dataset = FlickrDataset(
        csv_file=r"data\captions.csv",
        root_dir=r"data\Flicker8k_Dataset",  # Directory containing your images
        vocabulary=vocab,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to fixed size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    )
    
    # Create data loader
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # Adjust based on your CPU cores
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    # Create model
    model = ImageCaptioningModel(embed_size, hidden_size, len(vocab), num_layers).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<PAD>'])
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    for epoch in range(num_epochs):
        train_loss = train_model(train_loader, model, criterion, optimizer, device)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}')
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
        }, f'checkpoints/model_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    main()