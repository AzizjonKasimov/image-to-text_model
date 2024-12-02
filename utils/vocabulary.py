import re
from collections import Counter
import json
import pickle
import os

def build_vocabulary(descriptions, min_word_freq=5, save_path='data\vocabulary', save_format='json'):
    """
    Build a vocabulary from text descriptions and optionally save it to a file.
    
    Args:
        descriptions (list): List of text descriptions
        min_word_freq (int): Minimum frequency for a word to be included in vocabulary
        save_path (str): Path where to save the vocabulary. If None, vocabulary won't be saved
        save_format (str): Format to save vocabulary ('json' or 'pickle')
    
    Returns:
        dict: The vocabulary mapping words to indices
    """
    # 1. Preprocess all descriptions
    word_counts = Counter()
    for desc in descriptions:
        # Convert to lowercase and split into words
        words = re.sub(r'[^\w\s]', '', desc.lower()).split()
        word_counts.update(words)
    
    # 2. Create vocabulary with words appearing more than min_freq times
    vocabulary = {
        '<PAD>': 0,  # Padding token
        '<START>': 1,  # Start of sentence token
        '<END>': 2,    # End of sentence token
        '<UNK>': 3     # Unknown word token
    }
    
    # Add words that appear more than min_word_freq times
    idx = len(vocabulary)
    for word, count in word_counts.items():
        if count >= min_word_freq:
            vocabulary[word] = idx
            idx += 1
    
    # 3. Save vocabulary if save_path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        if save_format.lower() == 'json':
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(vocabulary, f, ensure_ascii=False, indent=2)
        elif save_format.lower() == 'pickle':
            with open(save_path, 'wb') as f:
                pickle.dump(vocabulary, f)
        else:
            raise ValueError("save_format must be either 'json' or 'pickle'")
    
    return vocabulary

def load_vocabulary(load_path='data\vocabulary', load_format='json'):
    """
    Load a previously saved vocabulary.
    
    Args:
        load_path (str): Path to the saved vocabulary file
        load_format (str): Format of the saved vocabulary ('json' or 'pickle')
    
    Returns:
        dict: The loaded vocabulary
    """
    if load_format.lower() == 'json':
        with open(load_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif load_format.lower() == 'pickle':
        with open(load_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError("load_format must be either 'json' or 'pickle'")


def text_to_sequence(text, vocabulary, max_length=50):
    # Convert text to sequence of indices
    words = re.sub(r'[^\w\s]', '', text.lower()).split()
    sequence = [vocabulary['<START>']]
    
    for word in words:
        if word in vocabulary:
            sequence.append(vocabulary[word])
        else:
            sequence.append(vocabulary['<UNK>'])
    
    sequence.append(vocabulary['<END>'])
    
    # Pad sequence to max_length
    if len(sequence) < max_length:
        sequence += [vocabulary['<PAD>']] * (max_length - len(sequence))
    else:
        sequence = sequence[:max_length]
    
    return sequence