import numpy as np
from collections import Counter
from typing import List, Dict, Tuple, Optional
import math


class Word2Vec:
    """
    Word2Vec-like embeddings implemented from scratch
    Supports Skip-gram and CBOW models
    """
    
    def __init__(self, embedding_dim: int = 100, window_size: int = 5,
                 model_type: str = 'skipgram', learning_rate: float = 0.01):
        """
        Initialize Word2Vec model
        
        Parameters:
        -----------
        embedding_dim : int
            Dimension of word embeddings
        window_size : int
            Context window size
        model_type : str
            'skipgram' or 'cbow'
        learning_rate : float
            Learning rate for training
        """
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.model_type = model_type
        self.learning_rate = learning_rate
        
        self.vocab = {}
        self.vocab_size = 0
        self.word_embeddings = None  # W matrix
        self.context_embeddings = None  # W' matrix
    
    def build_vocab(self, texts: List[str], min_count: int = 5):
        """
        Build vocabulary from texts
        
        Parameters:
        -----------
        texts : list
            List of text documents
        min_count : int
            Minimum word frequency
        """
        word_freq = Counter()
        
        for text in texts:
            words = text.lower().split()
            word_freq.update(words)
        
        # Filter by minimum count
        vocab_words = [word for word, count in word_freq.items() if count >= min_count]
        
        # Build vocab mapping
        self.vocab = {word: idx for idx, word in enumerate(sorted(vocab_words))}
        self.vocab_size = len(self.vocab)
        
        # Initialize embedding matrices
        self.word_embeddings = np.random.randn(self.vocab_size, self.embedding_dim) * 0.01
        self.context_embeddings = np.random.randn(self.vocab_size, self.embedding_dim) * 0.01
    
    def train(self, texts: List[str], epochs: int = 5, negative_samples: int = 15):
        """
        Train word embeddings
        
        Parameters:
        -----------
        texts : list
            List of text documents
        epochs : int
            Number of training epochs
        negative_samples : int
            Number of negative samples
        """
        if self.vocab_size == 0:
            self.build_vocab(texts)
        
        # Prepare training data
        sentences = [text.lower().split() for text in texts]
        
        for epoch in range(epochs):
            for sentence in sentences:
                # Filter out unknown words
                words = [w for w in sentence if w in self.vocab]
                
                for pos, word in enumerate(words):
                    word_idx = self.vocab[word]
                    
                    # Define context window
                    start = max(0, pos - self.window_size)
                    end = min(len(words), pos + self.window_size + 1)
                    
                    context_words = words[start:end]
                    context_words.remove(word)  # Remove center word
                    
                    if self.model_type == 'skipgram':
                        self._train_skipgram(word_idx, context_words, negative_samples)
                    else:
                        self._train_cbow(word_idx, context_words)
    
    def _train_skipgram(self, word_idx: int, context_words: List[str], 
                       negative_samples: int):
        """Train skip-gram model"""
        # Positive samples
        for context_word in context_words:
            context_idx = self.vocab[context_word]
            
            # Forward pass
            dot_product = np.dot(self.word_embeddings[word_idx],
                               self.context_embeddings[context_idx])
            pred = 1.0 / (1.0 + np.exp(-dot_product))  # Sigmoid
            
            # Backward pass
            error = (1 - pred)
            delta = error * self.context_embeddings[context_idx]
            
            self.word_embeddings[word_idx] += self.learning_rate * delta
            self.context_embeddings[context_idx] += self.learning_rate * error * \
                                                    self.word_embeddings[word_idx]
        
        # Negative samples
        for _ in range(negative_samples):
            random_idx = np.random.randint(self.vocab_size)
            
            dot_product = np.dot(self.word_embeddings[word_idx],
                               self.context_embeddings[random_idx])
            pred = 1.0 / (1.0 + np.exp(-dot_product))
            
            error = -pred
            delta = error * self.context_embeddings[random_idx]
            
            self.word_embeddings[word_idx] += self.learning_rate * delta
            self.context_embeddings[random_idx] += self.learning_rate * error * \
                                                   self.word_embeddings[word_idx]
    
    def _train_cbow(self, word_idx: int, context_words: List[str]):
        """Train CBOW model"""
        if not context_words:
            return
        
        # Average context word embeddings
        context_indices = [self.vocab[w] for w in context_words if w in self.vocab]
        if not context_indices:
            return
        
        context_embeddings_avg = np.mean(
            self.context_embeddings[context_indices], axis=0
        )
        
        # Forward pass
        dot_product = np.dot(context_embeddings_avg,
                            self.word_embeddings[word_idx])
        pred = 1.0 / (1.0 + np.exp(-dot_product))
        
        # Backward pass
        error = (1 - pred)
        delta = error * self.word_embeddings[word_idx]
        
        for context_idx in context_indices:
            self.context_embeddings[context_idx] += self.learning_rate * delta
    
    def get_embedding(self, word: str) -> Optional[np.ndarray]:
        """
        Get embedding for a word
        
        Parameters:
        -----------
        word : str
            Word to get embedding for
        
        Returns:
        --------
        ndarray : Word embedding
        """
        word = word.lower()
        if word in self.vocab:
            return self.word_embeddings[self.vocab[word]].copy()
        return None
    
    def most_similar(self, word: str, topn: int = 10) -> List[Tuple[str, float]]:
        """
        Find most similar words
        
        Parameters:
        -----------
        word : str
            Query word
        topn : int
            Number of similar words to return
        
        Returns:
        --------
        list : List of (word, similarity) tuples
        """
        embedding = self.get_embedding(word)
        if embedding is None:
            return []
        
        # Normalize query embedding
        embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
        
        # Compute similarities
        similarities = []
        for vocab_word, idx in self.vocab.items():
            if vocab_word == word:
                continue
            
            word_emb = self.word_embeddings[idx]
            word_emb = word_emb / (np.linalg.norm(word_emb) + 1e-10)
            
            similarity = np.dot(embedding, word_emb)
            similarities.append((vocab_word, similarity))
        
        # Sort and return top N
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:topn]
    
    def word_distance(self, word1: str, word2: str) -> float:
        """
        Compute cosine distance between two words
        
        Parameters:
        -----------
        word1 : str
            First word
        word2 : str
            Second word
        
        Returns:
        --------
        float : Cosine similarity
        """
        emb1 = self.get_embedding(word1)
        emb2 = self.get_embedding(word2)
        
        if emb1 is None or emb2 is None:
            return 0.0
        
        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm_product = np.linalg.norm(emb1) * np.linalg.norm(emb2)
        
        return dot_product / (norm_product + 1e-10)
    
    def save_embeddings(self, filepath: str):
        """Save embeddings to file"""
        np.savez(filepath,
                embeddings=self.word_embeddings,
                vocab=list(self.vocab.keys()))
    
    def load_embeddings(self, filepath: str):
        """Load embeddings from file"""
        data = np.load(filepath)
        self.word_embeddings = data['embeddings']
        vocab_words = data['vocab'].tolist()
        self.vocab = {word: idx for idx, word in enumerate(vocab_words)}
        self.vocab_size = len(self.vocab)


class FastTextEmbeddings:
    """
    FastText-like embeddings
    Uses subword information for better OOV handling
    """
    
    def __init__(self, embedding_dim: int = 100, min_n: int = 3, max_n: int = 6):
        """
        Initialize FastText model
        
        Parameters:
        -----------
        embedding_dim : int
            Embedding dimension
        min_n : int
            Minimum n-gram size
        max_n : int
            Maximum n-gram size
        """
        self.embedding_dim = embedding_dim
        self.min_n = min_n
        self.max_n = max_n
        
        self.word_vectors = {}
        self.ngram_vectors = {}
    
    def _get_ngrams(self, word: str) -> List[str]:
        """
        Extract n-grams from word
        
        Parameters:
        -----------
        word : str
            Word to extract n-grams from
        
        Returns:
        --------
        list : List of n-grams
        """
        # Add boundary markers
        extended_word = f'#{word}#'
        ngrams = []
        
        for n in range(self.min_n, self.max_n + 1):
            for i in range(len(extended_word) - n + 1):
                ngrams.append(extended_word[i:i+n])
        
        return ngrams
    
    def get_vector(self, word: str) -> np.ndarray:
        """
        Get embedding vector for word using n-grams
        Falls back to n-gram based OOV handling
        """
        if word in self.word_vectors:
            return self.word_vectors[word].copy()
        
        # OOV handling: average n-gram vectors
        ngrams = self._get_ngrams(word)
        vectors = []
        
        for ngram in ngrams:
            if ngram in self.ngram_vectors:
                vectors.append(self.ngram_vectors[ngram])
        
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.random.randn(self.embedding_dim) * 0.01


class GloVeEmbeddings:
    """
    GloVe-like embeddings (Global Vectors)
    Uses word co-occurrence matrix
    """
    
    def __init__(self, embedding_dim: int = 100, window_size: int = 5):
        """Initialize GloVe model"""
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.vocab = {}
        self.embeddings = None
    
    def build_cooccurrence_matrix(self, texts: List[str]) -> np.ndarray:
        """
        Build word co-occurrence matrix
        
        Parameters:
        -----------
        texts : list
            List of text documents
        
        Returns:
        --------
        ndarray : Co-occurrence matrix
        """
        # Build vocabulary
        word_freq = Counter()
        for text in texts:
            words = text.lower().split()
            word_freq.update(words)
        
        self.vocab = {word: idx for idx, (word, _) in 
                     enumerate(sorted(word_freq.items(), key=lambda x: x[1], reverse=True))}
        vocab_size = len(self.vocab)
        
        # Build co-occurrence matrix
        cooccurrence = np.zeros((vocab_size, vocab_size))
        
        for text in texts:
            words = text.lower().split()
            
            for i, word in enumerate(words):
                if word not in self.vocab:
                    continue
                
                word_idx = self.vocab[word]
                
                # Context window
                start = max(0, i - self.window_size)
                end = min(len(words), i + self.window_size + 1)
                
                for j in range(start, end):
                    if i != j and words[j] in self.vocab:
                        context_idx = self.vocab[words[j]]
                        cooccurrence[word_idx, context_idx] += 1
        
        return cooccurrence
    
    def get_embedding(self, word: str) -> Optional[np.ndarray]:
        """Get word embedding"""
        word = word.lower()
        if word in self.vocab and self.embeddings is not None:
            return self.embeddings[self.vocab[word]].copy()
        return None
