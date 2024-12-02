import torch
from torchvision import transforms
from PIL import Image
from models.model import ImageCaptioningModel
import json

class CaptionInference:
    def __init__(self, model_path, vocab_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        # Load vocabulary
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
        
        # Initialize model
        self.embed_size = 256  # Make sure these match your training parameters
        self.hidden_size = 512
        self.vocab_size = len(self.vocab)
        
        # Create and load model
        self.model = ImageCaptioningModel(
            embed_size=self.embed_size,
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size
        ).to(self.device)
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def idx_to_word(self, idx):
        for word, index in self.vocab.items():
            if index == idx:
                return word
        return self.vocab['<UNK>']
    
    def generate_caption(self, image_path, max_length=20):
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        # Generate caption
        with torch.no_grad():
            caption_indices = self.model.generate_caption(
                image,
                self.vocab,
                max_length=max_length
            )
        
        # Convert indices to words
        caption_words = []
        for idx in caption_indices:
            word = self.idx_to_word(idx)
            if word == '<END>':
                break
            if word not in ['<START>', '<PAD>', '<UNK>']:
                caption_words.append(word)
        
        return ' '.join(caption_words)

# Example usage
def main():
    # Initialize inference
    inference = CaptionInference(
        model_path='checkpoints/model_epoch_2.pth',  # Update with your model path
        vocab_path='vocabulary.json'  # Update with your vocabulary path
    )
    
    # Generate caption for a single image
    image_path = 'path/to/your/test/image.jpg'  # Update with your image path
    caption = inference.generate_caption(image_path)
    print(f"Generated caption: {caption}")
    
    # Generate captions for multiple images
    test_images = [
        'path/to/image1.jpg',
        'path/to/image2.jpg',
        'path/to/image3.jpg'
    ]
    
    for img_path in test_images:
        caption = inference.generate_caption(img_path)
        print(f"\nImage: {img_path}")
        print(f"Caption: {caption}")

if __name__ == '__main__':
    main()