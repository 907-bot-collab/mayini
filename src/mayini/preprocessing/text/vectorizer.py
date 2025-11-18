import numpy as np
from collections import Counter
from typing import List, Dict, Tuple, Optional
import math


class TFIDFVectorizer:
    """
    TF-IDF Vectorizer implemented from scratch with NumPy
    
    TF-IDF = Term Frequency × Inverse Document Frequency
    TF = count(term) / len(document)
    IDF = log(n_documents / n_documents_with_term)
    """
    
    def __init__(self, max_features: int = 5000, lowercase: bool = True,
                 min_df: int = 1, max_df: float = 1.0, ngram_range: Tuple = (1, 1)):
        """
        Initialize TF-IDF Vectorizer
        
        Parameters:
        -----------
        max_features : int
            Maximum number of features (vocabulary size)
        lowercase : bool
            Convert text to lowercase
        min_df : int
            Minimum document frequency (ignore terms in fewer documents)
        max_df : float
            Maximum document frequency (ignore very common terms)
        ngram_range : tuple
            Range of n-grams (min_n, max_n)
        """
        self.max_features = max_features
        self.lowercase = lowercase
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        
        self.vocabulary = {}
        self.idf_scores = None
        self.feature_names = None
        self.n_documents = 0
    
    def fit(self, documents: List[str]) -> 'TFIDFVectorizer':
        """
        Fit the vectorizer on a corpus of documents
        
        Parameters:
        -----------
        documents : list
            List of text documents
        
        Returns:
        --------
        self : Fitted vectorizer
        """
        self.n_documents = len(documents)
        
        # Step 1: Tokenize and build vocabulary
        doc_term_freq = []
        term_doc_freq = Counter()
        
        for doc in documents:
            tokens = self._tokenize(doc)
            doc_term_freq.append(tokens)
            
            # Count unique terms per document
            unique_terms = set(tokens)
            term_doc_freq.update(unique_terms)
        
        # Step 2: Filter by document frequency
        filtered_vocab = {}
        for term, freq in term_doc_freq.items():
            if freq >= self.min_df and freq / self.n_documents <= self.max_df:
                filtered_vocab[term] = freq
        
        # Step 3: Select top features by frequency
        sorted_terms = sorted(filtered_vocab.items(), 
                             key=lambda x: x[1], reverse=True)
        
        top_terms = sorted_terms[:self.max_features]
        
        # Step 4: Build vocabulary mapping
        self.vocabulary = {term: idx for idx, (term, _) in enumerate(top_terms)}
        self.feature_names = [term for term, _ in top_terms]
        
        # Step 5: Compute IDF scores
        self._compute_idf(doc_term_freq)
        
        return self
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into n-grams
        
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
        
        # Simple word tokenization
        tokens = text.split()
        
        # Generate n-grams
        ngrams = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(tokens) - n + 1):
                ngram = ' '.join(tokens[i:i+n])
                ngrams.append(ngram)
        
        return ngrams
    
    def _compute_idf(self, doc_term_freq: List[List[str]]):
        """
        Compute IDF scores for all vocabulary terms
        
        IDF = log(n_documents / n_documents_with_term)
        """
        vocab_size = len(self.vocabulary)
        self.idf_scores = np.zeros(vocab_size)
        
        # Count document frequency for each term
        doc_freq = np.zeros(vocab_size)
        
        for tokens in doc_term_freq:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                if token in self.vocabulary:
                    idx = self.vocabulary[token]
                    doc_freq[idx] += 1
        
        # Compute IDF
        for idx in range(vocab_size):
            if doc_freq[idx] > 0:
                self.idf_scores[idx] = math.log(self.n_documents / doc_freq[idx])
            else:
                self.idf_scores[idx] = 0
    
    def transform(self, documents: List[str]) -> np.ndarray:
        """
        Transform documents to TF-IDF matrix
        
        Parameters:
        -----------
        documents : list
            List of text documents
        
        Returns:
        --------
        ndarray : TF-IDF matrix (n_documents x n_features)
        """
        n_docs = len(documents)
        vocab_size = len(self.vocabulary)
        
        tfidf_matrix = np.zeros((n_docs, vocab_size), dtype=np.float32)
        
        for doc_idx, doc in enumerate(documents):
            tokens = self._tokenize(doc)
            
            # Step 1: Compute term frequencies
            term_freq = Counter(tokens)
            
            # Step 2: Compute TF-IDF
            for term, count in term_freq.items():
                if term in self.vocabulary:
                    term_idx = self.vocabulary[term]
                    
                    # TF = count / document_length
                    tf = count / len(tokens) if len(tokens) > 0 else 0
                    
                    # TF-IDF = TF * IDF
                    tfidf = tf * self.idf_scores[term_idx]
                    
                    tfidf_matrix[doc_idx, term_idx] = tfidf
        
        # Step 3: L2 Normalization
        norms = np.sqrt((tfidf_matrix ** 2).sum(axis=1, keepdims=True))
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        tfidf_matrix = tfidf_matrix / norms
        
        return tfidf_matrix
    
    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """
        Fit vectorizer and transform documents in one step
        
        Parameters:
        -----------
        documents : list
            List of text documents
        
        Returns:
        --------
        ndarray : TF-IDF matrix
        """
        self.fit(documents)
        return self.transform(documents)
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names (vocabulary)"""
        return self.feature_names if self.feature_names is not None else []
    
    def get_vocabulary(self) -> Dict[str, int]:
        """Get vocabulary mapping"""
        return self.vocabulary.copy()


class CountVectorizer:
    """
    Count Vectorizer - Simpler than TF-IDF
    Just counts term occurrences in each document
    """
    
    def __init__(self, max_features: int = 5000, lowercase: bool = True):
        """Initialize count vectorizer"""
        self.max_features = max_features
        self.lowercase = lowercase
        self.vocabulary = {}
        self.feature_names = None
    
    def fit(self, documents: List[str]) -> 'CountVectorizer':
        """Fit on documents"""
        term_freq = Counter()
        
        for doc in documents:
            if self.lowercase:
                doc = doc.lower()
            
            tokens = doc.split()
            term_freq.update(tokens)
        
        # Select top features
        top_terms = term_freq.most_common(self.max_features)
        
        self.vocabulary = {term: idx for idx, (term, _) in enumerate(top_terms)}
        self.feature_names = [term for term, _ in top_terms]
        
        return self
    
    def transform(self, documents: List[str]) -> np.ndarray:
        """Transform to count matrix"""
        n_docs = len(documents)
        vocab_size = len(self.vocabulary)
        
        count_matrix = np.zeros((n_docs, vocab_size), dtype=np.int32)
        
        for doc_idx, doc in enumerate(documents):
            if self.lowercase:
                doc = doc.lower()
            
            tokens = doc.split()
            term_freq = Counter(tokens)
            
            for term, count in term_freq.items():
                if term in self.vocabulary:
                    term_idx = self.vocabulary[term]
                    count_matrix[doc_idx, term_idx] = count
        
        return count_matrix
    
    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """Fit and transform"""
        self.fit(documents)
        return self.transform(documents)


class BinaryVectorizer:
    """
    Binary Vectorizer - 0/1 representation
    Indicates presence/absence of terms
    """
    
    def __init__(self, max_features: int = 5000, lowercase: bool = True):
        """Initialize binary vectorizer"""
        self.max_features = max_features
        self.lowercase = lowercase
        self.vocabulary = {}
        self.feature_names = None
    
    def fit(self, documents: List[str]) -> 'BinaryVectorizer':
        """Fit on documents"""
        term_freq = Counter()
        
        for doc in documents:
            if self.lowercase:
                doc = doc.lower()
            
            unique_terms = set(doc.split())
            term_freq.update(unique_terms)
        
        # Select top features
        top_terms = term_freq.most_common(self.max_features)
        
        self.vocabulary = {term: idx for idx, (term, _) in enumerate(top_terms)}
        self.feature_names = [term for term, _ in top_terms]
        
        return self
    
    def transform(self, documents: List[str]) -> np.ndarray:
        """Transform to binary matrix"""
        n_docs = len(documents)
        vocab_size = len(self.vocabulary)
        
        binary_matrix = np.zeros((n_docs, vocab_size), dtype=np.int8)
        
        for doc_idx, doc in enumerate(documents):
            if self.lowercase:
                doc = doc.lower()
            
            unique_terms = set(doc.split())
            
            for term in unique_terms:
                if term in self.vocabulary:
                    term_idx = self.vocabulary[term]
                    binary_matrix[doc_idx, term_idx] = 1
        
        return binary_matrix
    
    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """Fit and transform"""
        self.fit(documents)
        return self.transform(documents)
