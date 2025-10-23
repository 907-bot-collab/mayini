import numpy as np
from collections import Counter
import re
from ..base import BaseTransformer


class CountVectorizer(BaseTransformer):
    """
    Convert text documents to token count matrix

    Parameters:
    -----------
    max_features : int, default=None
        Maximum number of features (vocabulary size)
    lowercase : bool, default=True
        Convert all text to lowercase

    Example:
    --------
    >>> from mayini.preprocessing import CountVectorizer
    >>> vectorizer = CountVectorizer(max_features=10)
    >>> docs = ["hello world", "hello python world"]
    >>> vectorizer.fit_transform(docs)
    """

    def __init__(self, max_features=None, lowercase=True):
        super().__init__()
        self.max_features = max_features
        self.lowercase = lowercase
        self.vocabulary_ = None

    def _tokenize(self, text):
        """Simple tokenization"""
        if self.lowercase:
            text = text.lower()
        tokens = re.findall(r'\w+', text)
        return tokens

    def fit(self, X, y=None):
        """Build vocabulary"""
        word_counts = Counter()

        for doc in X:
            tokens = self._tokenize(doc)
            word_counts.update(tokens)

        if self.max_features is not None:
            most_common = word_counts.most_common(self.max_features)
        else:
            most_common = word_counts.items()

        self.vocabulary_ = {word: idx for idx, (word, _) in enumerate(most_common)}
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Transform documents to count matrix"""
        self._check_is_fitted()

        n_docs = len(X)
        n_features = len(self.vocabulary_)
        X_counts = np.zeros((n_docs, n_features))

        for i, doc in enumerate(X):
            tokens = self._tokenize(doc)
            token_counts = Counter(tokens)

            for word, count in token_counts.items():
                if word in self.vocabulary_:
                    X_counts[i, self.vocabulary_[word]] = count

        return X_counts


class TfidfVectorizer(BaseTransformer):
    """
    Convert text documents to TF-IDF features

    TF-IDF = Term Frequency * Inverse Document Frequency

    Parameters:
    -----------
    max_features : int, default=None
        Maximum number of features
    lowercase : bool, default=True
        Convert to lowercase
    use_idf : bool, default=True
        Enable IDF reweighting

    Example:
    --------
    >>> from mayini.preprocessing import TfidfVectorizer
    >>> vectorizer = TfidfVectorizer(max_features=10)
    >>> docs = ["hello world", "hello python", "python world"]
    >>> vectorizer.fit_transform(docs)
    """

    def __init__(self, max_features=None, lowercase=True, use_idf=True):
        super().__init__()
        self.max_features = max_features
        self.lowercase = lowercase
        self.use_idf = use_idf
        self.vocabulary_ = None
        self.idf_ = None

    def _tokenize(self, text):
        """Simple tokenization"""
        if self.lowercase:
            text = text.lower()
        tokens = re.findall(r'\w+', text)
        return tokens

    def fit(self, X, y=None):
        """Build vocabulary and IDF"""
        word_doc_counts = Counter()
        all_words = Counter()

        for doc in X:
            tokens = self._tokenize(doc)
            unique_tokens = set(tokens)
            word_doc_counts.update(unique_tokens)
            all_words.update(tokens)

        if self.max_features is not None:
            most_common = all_words.most_common(self.max_features)
        else:
            most_common = all_words.items()

        self.vocabulary_ = {word: idx for idx, (word, _) in enumerate(most_common)}

        if self.use_idf:
            n_docs = len(X)
            self.idf_ = np.zeros(len(self.vocabulary_))

            for word, idx in self.vocabulary_.items():
                doc_freq = word_doc_counts.get(word, 0)
                self.idf_[idx] = np.log((n_docs + 1) / (doc_freq + 1)) + 1

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Transform documents to TF-IDF matrix"""
        self._check_is_fitted()

        n_docs = len(X)
        n_features = len(self.vocabulary_)
        X_tfidf = np.zeros((n_docs, n_features))

        for i, doc in enumerate(X):
            tokens = self._tokenize(doc)
            token_counts = Counter(tokens)

            # Compute TF
            total_tokens = len(tokens)
            if total_tokens == 0:
                continue

            for word, count in token_counts.items():
                if word in self.vocabulary_:
                    idx = self.vocabulary_[word]
                    tf = count / total_tokens

                    if self.use_idf:
                        X_tfidf[i, idx] = tf * self.idf_[idx]
                    else:
                        X_tfidf[i, idx] = tf

        return X_tfidf
