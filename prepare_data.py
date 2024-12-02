# prepare_data.py
import os
from utils.download import download_dataset
from utils.preprocess import build_vocabulary
import pandas as pd



def prepare_data():
    # Download dataset if not already present
    if not os.path.exists('data/Flickr8k_Dataset'):
        download_dataset()
    
    # Process captions file
    captions_file = 'data/Flickr8k.lemma.token.txt'
    
    # Read and process captions
    with open(captions_file, 'r') as f:
        captions = f.read().strip().split('\n')
    
    # Create DataFrame
    data = []
    for caption in captions:
        img_caption = caption.split('\t')
        img_name = img_caption[0].split('#')[0]
        caption_text = img_caption[1]
        data.append([img_name, caption_text])
    
    df = pd.DataFrame(data, columns=['image', 'caption'])
    
    # Save processed captions
    df.to_csv('data/captions.csv', index=False, sep='|')
    
    # Build vocabulary
    vocab = build_vocabulary('data/captions.csv')
    
    # Save vocabulary size
    print(f"Vocabulary size: {len(vocab)}")
    
    return vocab

if __name__ == "__main__":
    prepare_data()