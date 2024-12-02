import torch
import torch.nn as nn

class CaptionDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(CaptionDecoder, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, features, captions):
        # Remove end token from captions
        embeddings = self.embed(captions[:, :-1])
        
        # Concatenate image features and caption embeddings
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        
        # LSTM forward pass
        hidden, _ = self.lstm(embeddings)
        outputs = self.linear(self.dropout(hidden))
        
        return outputs
    
    def generate_caption(self, features, vocab, max_length=20):
        batch_size = features.size(0)
        device = features.device
        
        # Initialize hidden states
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        hidden = (h, c)
        
        # Initialize input as image features
        inputs = self.embed(torch.tensor([vocab['<START>']]).to(device)).unsqueeze(1)
        
        captions = []
        
        for _ in range(max_length):
            # Get output from LSTM
            lstm_out, hidden = self.lstm(inputs, hidden)
            outputs = self.linear(lstm_out.squeeze(1))
            predicted = outputs.argmax(1)
            
            captions.append(predicted.item())
            
            # Break if end token is predicted
            if predicted.item() == vocab['<END>']:
                break
                
            # Prepare input for next iteration
            inputs = self.embed(predicted).unsqueeze(1)
            
        return captions