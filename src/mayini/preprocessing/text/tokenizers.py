
import re
import numpy as np
from collections import Counter
from typing import List, Tuple, Dict, Set, Optional


class Tokenizer:
    """
    Universal Tokenizer supporting multiple tokenization strategies
    - Word tokenization
    - Character tokenization
    - Subword tokenization (BPE-like)
    """
    
    def __init__(self, tokenization_type: str = 'word', lowercase: bool = True):
        """
        Initialize tokenizer
        
        Parameters:
        -----------
        tokenization_type : str
            Type of tokenization: 'word', 'character', or 'subword'
        lowercase : bool
            Whether to convert text to lowercase
        """
        self.tokenization_type = tokenization_type
        self.lowercase = lowercase
        self.vocab = {}
        self.idx_to_token = {}
        self.token_to_idx = {}
        self.vocab_size = 0
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text based on selected strategy
        
        Parameters:
        -----------
        text : str
            Text to tokenize
        
        Returns:
        --------
        list : List of tokens
        """
        if self.lowercase:
            text = text.lower()
        
        if self.tokenization_type == 'word':
            return self._word_tokenize(text)
        elif self.tokenization_type == 'character':
            return self._character_tokenize(text)
        elif self.tokenization_type == 'subword':
            return self._subword_tokenize(text)
        else:
            raise ValueError(f"Unknown tokenization type: {self.tokenization_type}")
    
    def _word_tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words
        Removes punctuation and splits on whitespace
        """
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Split on word boundaries and punctuation
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
        
        # Filter out empty tokens
        tokens = [t for t in tokens if t.strip()]
        
        return tokens
    
    def _character_tokenize(self, text: str) -> List[str]:
        """Tokenize text into individual characters"""
        # Remove whitespace
        text = text.replace(' ', '')
        return list(text)
    
    def _subword_tokenize(self, text: str) -> List[str]:
        """
        Simple subword tokenization (BPE-like)
        Splits into word pieces preserving boundaries
        """
        # First, split into words
        words = text.split()
        
        subwords = []
        for word in words:
            # Keep word boundaries
            if word:
                subwords.append('##' + word if subwords else word)
        
        return subwords
    
    def build_vocab(self, texts: List[str], max_vocab_size: int = 10000,
                   min_freq: int = 1) -> Dict[str, int]:
        """
        Build vocabulary from list of texts
        
        Parameters:
        -----------
        texts : list
            List of text strings
        max_vocab_size : int
            Maximum vocabulary size
        min_freq : int
            Minimum token frequency to include
        
        Returns:
        --------
        dict : Vocabulary mapping {token: index}
        """
        # Count token frequencies
        token_freq = Counter()
        
        for text in texts:
            tokens = self.tokenize(text)
            token_freq.update(tokens)
        
        # Filter by minimum frequency
        filtered_tokens = {token: freq for token, freq in token_freq.items() 
                          if freq >= min_freq}
        
        # Sort by frequency and select top tokens
        sorted_tokens = sorted(filtered_tokens.items(), 
                              key=lambda x: x[1], reverse=True)
        
        top_tokens = sorted_tokens[:max_vocab_size]
        
        # Create vocabulary mappings
        self.vocab = {token: idx for idx, (token, _) in enumerate(top_tokens)}
        self.idx_to_token = {idx: token for token, idx in self.vocab.items()}
        self.token_to_idx = self.vocab
        self.vocab_size = len(self.vocab)
        
        return self.vocab
    
    def encode(self, text: str, unknown_token: str = '<UNK>',
              pad_to_length: Optional[int] = None) -> np.ndarray:
        """
        Convert text to token indices
        
        Parameters:
        -----------
        text : str
            Text to encode
        unknown_token : str
            Token for unknown words
        pad_to_length : int, optional
            Pad output to this length
        
        Returns:
        --------
        ndarray : Array of token indices
        """
        tokens = self.tokenize(text)
        indices = []
        
        for token in tokens:
            if token in self.vocab:
                indices.append(self.vocab[token])
            else:
                # Handle unknown tokens
                if unknown_token in self.vocab:
                    indices.append(self.vocab[unknown_token])
        
        indices = np.array(indices, dtype=np.int32)
        
        # Pad or truncate
        if pad_to_length is not None:
            if len(indices) < pad_to_length:
                indices = np.pad(indices, (0, pad_to_length - len(indices)), 
                               mode='constant', constant_values=0)
            else:
                indices = indices[:pad_to_length]
        
        return indices
    
    def decode(self, indices: np.ndarray, skip_special_tokens: bool = True) -> str:
        """
        Convert token indices back to text
        
        Parameters:
        -----------
        indices : ndarray
            Array of token indices
        skip_special_tokens : bool
            Whether to skip special tokens in output
        
        Returns:
        --------
        str : Decoded text
        """
        tokens = []
        
        for idx in indices:
            idx = int(idx)
            if idx in self.idx_to_token:
                token = self.idx_to_token[idx]
                
                # Skip special tokens if requested
                if skip_special_tokens and (token.startswith('<') and token.endswith('>')):
                    continue
                
                tokens.append(token)
        
        # Join tokens with spaces, handling subword tokens
        text = ''
        for i, token in enumerate(tokens):
            if token.startswith('##'):
                text += token[2:]  # Remove ## prefix
            else:
                if i > 0:
                    text += ' '
                text += token
        
        return text.strip()
    
    def get_ngrams(self, text: str, n: int = 2) -> List[Tuple[str, ...]]:
        """
        Generate n-grams from text
        
        Parameters:
        -----------
        text : str
            Text to process
        n : int
            Size of n-grams
        
        Returns:
        --------
        list : List of n-gram tuples
        """
        tokens = self.tokenize(text)
        ngrams = []
        
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams.append(ngram)
        
        return ngrams


class CharacterTokenizer:
    """Character-level tokenization"""
    
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
    
    def build_vocab(self, text: str) -> Dict[str, int]:
        """
        Build character vocabulary from text
        
        Parameters:
        -----------
        text : str
            Text to extract characters from
        
        Returns:
        --------
        dict : Character to index mapping
        """
        unique_chars = sorted(set(text))
        self.char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
        
        return self.char_to_idx
    
    def encode(self, text: str, pad_to_length: Optional[int] = None) -> np.ndarray:
        """
        Encode text to character indices
        
        Parameters:
        -----------
        text : str
            Text to encode
        pad_to_length : int, optional
            Pad to this length
        
        Returns:
        --------
        ndarray : Array of character indices
        """
        indices = []
        for char in text:
            if char in self.char_to_idx:
                indices.append(self.char_to_idx[char])
        
        indices = np.array(indices, dtype=np.int32)
        
        # Pad if needed
        if pad_to_length is not None:
            if len(indices) < pad_to_length:
                indices = np.pad(indices, (0, pad_to_length - len(indices)), 
                               mode='constant', constant_values=0)
            else:
                indices = indices[:pad_to_length]
        
        return indices
    
    def decode(self, indices: np.ndarray) -> str:
        """Decode character indices to text"""
        return ''.join([self.idx_to_char[int(idx)] for idx in indices 
                       if int(idx) in self.idx_to_char])


class WordPieceTokenizer:
    """
    WordPiece tokenization (similar to BERT tokenization)
    Builds vocabulary from most frequent subword pieces
    """
    
    def __init__(self, vocab_size: int = 30000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.idx_to_token = {}
    
    def build_vocab(self, texts: List[str]):
        """
        Build WordPiece vocabulary
        
        Parameters:
        -----------
        texts : list
            List of texts to build vocabulary from
        """
        # Initialize with character tokens
        char_freq = Counter()
        
        for text in texts:
            for char in text.lower():
                char_freq[char] += 1
        
        # Start with characters
        vocab_tokens = list(char_freq.keys())
        
        # Iteratively merge most frequent subword pairs
        vocab_idx = 0
        self.vocab = {token: vocab_idx for vocab_idx, token in enumerate(vocab_tokens)}
        self.idx_to_token = {idx: token for token, idx in self.vocab.items()}
    
    def encode(self, text: str) -> np.ndarray:
        """
        Encode text using WordPiece vocabulary
        
        Parameters:
        -----------
        text : str
            Text to encode
        
        Returns:
        --------
        ndarray : Encoded indices
        """
        text = text.lower()
        tokens = []
        
        i = 0
        while i < len(text):
            # Try to find longest matching token
            found = False
            
            for j in range(len(text), i, -1):
                subword = text[i:j]
                
                if subword in self.vocab:
                    tokens.append(self.vocab[subword])
                    i = j
                    found = True
                    break
            
            if not found:
                # Unknown character, skip
                i += 1
        
        return np.array(tokens, dtype=np.int32)
