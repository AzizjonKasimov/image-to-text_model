from torch.utils.data import Dataset
from PIL import Image
import torch
from utils.vocabulary import text_to_sequence
import os
import pandas as pd

class FlickrDataset(Dataset):
    def __init__(self, csv_file, root_dir, vocabulary, transform=None, max_length=50):
        """
        Args:
            csv_file (string): Path to the csv file with captions
            root_dir (string): Directory with all the images
            vocabulary (dict): Vocabulary dictionary
            transform (callable, optional): Optional transform to be applied on images
            max_length (int): Maximum length of the caption
        """
        self.df = pd.read_csv(csv_file, delimiter='|')
        self.root_dir = root_dir
        self.vocabulary = vocabulary
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image path
        img_name = self.df.iloc[idx]['image']
        image_path = os.path.join(self.root_dir, img_name)
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        # Get caption and convert to tensor
        caption = self.df.iloc[idx]['caption']
        caption_sequence = text_to_sequence(caption, self.vocabulary, self.max_length)
        caption_tensor = torch.tensor(caption_sequence)
        
        return image, caption_tensor