import torch

# configs/config.py
class Config:
    # Data parameters
    data_dir = "data"
    train_folder = "train"
    val_folder = "val"
    captions_file = "captions.txt"
    
    # Model parameters
    embed_size = 256
    hidden_size = 512
    num_layers = 1
    
    # Training parameters
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    
    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Logging
    log_dir = "logs"
    checkpoint_dir = "checkpoints"