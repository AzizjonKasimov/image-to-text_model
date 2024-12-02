# utils/download.py
import os
import urllib.request
import zipfile
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_dataset():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # URLs for Flickr8k dataset
    image_url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
    text_url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"
    
    # Download and extract images
    print("Downloading images...")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1) as t:
        urllib.request.urlretrieve(image_url, 'data/Flickr8k_Dataset.zip', reporthook=t.update_to)
    
    print("Extracting images...")
    with zipfile.ZipFile('data/Flickr8k_Dataset.zip', 'r') as zip_ref:
        zip_ref.extractall('data')
    
    # Download and extract text data
    print("Downloading text data...")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1) as t:
        urllib.request.urlretrieve(text_url, 'data/Flickr8k_text.zip', reporthook=t.update_to)
    
    print("Extracting text data...")
    with zipfile.ZipFile('data/Flickr8k_text.zip', 'r') as zip_ref:
        zip_ref.extractall('data')

if __name__ == "__main__":
    download_dataset()