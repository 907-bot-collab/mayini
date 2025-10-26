import numpy as np
from collections import Counter
from ..base import BaseTransformer


class CountVectorizer(BaseTransformer):
    """
    Convert text documents to matrix of token counts

    Parameters
    ----------
    max_features : int, default=None
        Maximum number of features (vocabulary size)
    min_df : int, default=1
        Minimum document frequency
    max_df : float, default=1.0
        Maximum document frequency (proportion)

    Example
    -------
    >>> from mayini.preprocessing import CountVectorizer
    >>> cv = CountVectorizer()
    >>> docs = ["hello world", "hello there"]
    >>> X = cv.fit_transform(docs)
    """

    def __init__(self, max_features=None, min_df=1, max_df=1.0):
        super().__init__()
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.vocabulary_ = None
        self.feature_names_ = None

    def fit(self, documents, y=None):
        """Build vocabulary"""
        # Count document frequency for each term
        doc_frequencies = Counter()
        term_counts = []

        for doc in documents:
            tokens = self._tokenize(doc)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_frequencies[token] += 1
            term_counts.append(Counter(tokens))

        n_docs = len(documents)

        # Filter by document frequency
        valid_terms = []
        for term, df in doc_frequencies.items():
            if df >= self.min_df and (df / n_docs) <= self.max_df:
                valid_terms.append((term, df))

        # Sort by document frequency and take top features
        valid_terms.sort(key=lambda x: x[1], reverse=True)

        if self.max_features:
            valid_terms = valid_terms[: self.max_features]

        # Build vocabulary
        self.vocabulary_ = {term: idx for idx, (term, _) in enumerate(valid_terms)}
        self.feature_names_ = [term for term, _ in valid_terms]

        self.is_fitted_ = True
        return self

    def transform(self, documents):
        """Transform documents to count matrix"""
        self._check_is_fitted()

        n_docs = len(documents)
        n_features = len(self.vocabulary_)
        X = np.zeros((n_docs, n_features))

        for doc_idx, doc in enumerate(documents):
            tokens = self._tokenize(doc)
            token_counts = Counter(tokens)

            for token, count in token_counts.items():
                if token in self.vocabulary_:
                    feature_idx = self.vocabulary_[token]
                    X[doc_idx, feature_idx] = count

        return X

    def get_feature_names(self):
        """Get feature names"""
        self._check_is_fitted()
        return self.feature_names_

    @staticmethod
    def _tokenize(document):
        """Simple tokenization"""
        # Convert to lowercase and split on whitespace
        return document.lower().split()


class TfidfVectorizer(BaseTransformer):
    """
    Convert text documents to TF-IDF feature matrix

    TF-IDF = Term Frequency * Inverse Document Frequency

    Parameters
    ----------
    max_features : int, default=None
        Maximum number of features
    min_df : int, default=1
        Minimum document frequency
    max_df : float, default=1.0
        Maximum document frequency
    use_idf : bool, default=True
        Enable inverse-document-frequency reweighting
    smooth_idf : bool, default=True
        Add one to document frequencies

    Example
    -------
    >>> from mayini.preprocessing import TfidfVectorizer
    >>> tfidf = TfidfVectorizer()
    >>> docs = ["hello world", "hello there"]
    >>> X = tfidf.fit_transform(docs)
    """

    def __init__(
        self,
        max_features=None,
        min_df=1,
        max_df=1.0,
        use_idf=True,
        smooth_idf=True,
    ):
        super().__init__()
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.vocabulary_ = None
        self.idf_ = None
        self.feature_names_ = None

    def fit(self, documents, y=None):
        """Build vocabulary and IDF"""
        # First, use CountVectorizer to build vocabulary
        self.count_vectorizer_ = CountVectorizer(
            max_features=self.max_features, min_df=self.min_df, max_df=self.max_df
        )
        count_matrix = self.count_vectorizer_.fit_transform(documents)

        self.vocabulary_ = self.count_vectorizer_.vocabulary_
        self.feature_names_ = self.count_vectorizer_.feature_names_

        # Compute IDF
        if self.use_idf:
            n_docs = len(documents)
            # Document frequency for each term
            df = np.sum(count_matrix > 0, axis=0)

            if self.smooth_idf:
                # Add one to document frequencies
                idf = np.log((n_docs + 1) / (df + 1)) + 1
            else:
                idf = np.log(n_docs / df) + 1

            self.idf_ = idf
        else:
            self.idf_ = np.ones(len(self.vocabulary_))

        self.is_fitted_ = True
        return self

    def transform(self, documents):
        """Transform documents to TF-IDF matrix"""
        self._check_is_fitted()

        # Get term frequency matrix
        tf_matrix = self.count_vectorizer_.transform(documents)

        # Apply IDF weighting
        tfidf_matrix = tf_matrix * self.idf_

        # L2 normalization (normalize each document vector)
        norms = np.sqrt(np.sum(tfidf_matrix**2, axis=1, keepdims=True))
        norms[norms == 0] = 1.0
        tfidf_matrix = tfidf_matrix / norms

        return tfidf_matrix

    def get_feature_names(self):
        """Get feature names"""
        self._check_is_fitted()
        return self.feature_names_
