
import re
import unicodedata
from typing import List, Set, Optional
import numpy as np


class TextCleaner:
    """
    Comprehensive text cleaning and normalization
    Handles multiple cleaning operations in one pipeline
    """
    
    def __init__(self):
        """Initialize cleaner with default stopwords"""
        self.stopwords = self._get_english_stopwords()
        self.custom_stopwords = set()
    
    def clean(self, text: str,
              remove_urls: bool = True,
              remove_emails: bool = True,
              remove_special_chars: bool = False,
              remove_numbers: bool = False,
              remove_accents: bool = True,
              remove_stopwords: bool = False,
              expand_contractions: bool = False,
              lowercase: bool = True,
              remove_extra_whitespace: bool = True) -> str:
        """
        Complete text cleaning pipeline
        
        Parameters:
        -----------
        text : str
            Text to clean
        remove_urls : bool
            Remove URL links
        remove_emails : bool
            Remove email addresses
        remove_special_chars : bool
            Remove special characters (keep alphanumeric + spaces)
        remove_numbers : bool
            Remove numeric values
        remove_accents : bool
            Remove diacritical marks
        remove_stopwords : bool
            Remove common English stopwords
        expand_contractions : bool
            Expand contractions (don't -> do not)
        lowercase : bool
            Convert to lowercase
        remove_extra_whitespace : bool
            Replace multiple spaces with single space
        
        Returns:
        --------
        str : Cleaned text
        """
        # Step 1: Remove URLs
        if remove_urls:
            text = self._remove_urls(text)
        
        # Step 2: Remove email addresses
        if remove_emails:
            text = self._remove_emails(text)
        
        # Step 3: Expand contractions (before other operations)
        if expand_contractions:
            text = self._expand_contractions(text)
        
        # Step 4: Remove accents
        if remove_accents:
            text = self._remove_accents(text)
        
        # Step 5: Remove special characters
        if remove_special_chars:
            text = self._remove_special_characters(text)
        
        # Step 6: Remove numbers
        if remove_numbers:
            text = self._remove_numbers(text)
        
        # Step 7: Lowercase
        if lowercase:
            text = text.lower()
        
        # Step 8: Remove extra whitespace
        if remove_extra_whitespace:
            text = self._remove_extra_whitespace(text)
        
        # Step 9: Remove stopwords
        if remove_stopwords:
            text = self._remove_stopwords(text)
        
        return text.strip()
    
    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text"""
        # Remove http/https/www URLs
        text = re.sub(r'http\S+|https\S+|www\S+', '', text)
        return text
    
    def _remove_emails(self, text: str) -> str:
        """Remove email addresses from text"""
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        return text
    
    def _remove_accents(self, text: str) -> str:
        """
        Remove accent marks from characters
        Uses Unicode NFD normalization
        """
        # Decompose characters with accents
        nfd = unicodedata.normalize('NFD', text)
        
        # Remove combining marks (accents)
        output = []
        for char in nfd:
            if unicodedata.category(char) != 'Mn':
                output.append(char)
        
        return ''.join(output)
    
    def _remove_special_characters(self, text: str) -> str:
        """
        Remove special characters, keep only alphanumeric and spaces
        """
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text
    
    def _remove_numbers(self, text: str) -> str:
        """Remove all numeric digits"""
        text = re.sub(r'\d+', '', text)
        return text
    
    def _remove_extra_whitespace(self, text: str) -> str:
        """Replace multiple whitespaces with single space"""
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def _remove_stopwords(self, text: str) -> str:
        """Remove English stopwords"""
        tokens = text.split()
        
        # Combine default and custom stopwords
        all_stopwords = self.stopwords.union(self.custom_stopwords)
        
        filtered = [t for t in tokens if t.lower() not in all_stopwords]
        return ' '.join(filtered)
    
    def _expand_contractions(self, text: str) -> str:
        """
        Expand English contractions
        Example: don't -> do not
        """
        contractions = {
            r"ain't": "am not",
            r"aren't": "are not",
            r"can't": "cannot",
            r"can't've": "cannot have",
            r"'cause": "because",
            r"could've": "could have",
            r"couldn't": "could not",
            r"didn't": "did not",
            r"doesn't": "does not",
            r"don't": "do not",
            r"hadn't": "had not",
            r"hasn't": "has not",
            r"haven't": "have not",
            r"he'd": "he would",
            r"he'll": "he will",
            r"he's": "he is",
            r"how'd": "how did",
            r"how'll": "how will",
            r"how's": "how is",
            r"i'd": "i would",
            r"i'll": "i will",
            r"i'm": "i am",
            r"i've": "i have",
            r"isn't": "is not",
            r"it'd": "it would",
            r"it'll": "it will",
            r"it's": "it is",
            r"let's": "let us",
            r"shouldn't": "should not",
            r"that's": "that is",
            r"there's": "there is",
            r"they'd": "they would",
            r"they'll": "they will",
            r"they're": "they are",
            r"they've": "they have",
            r"wasn't": "was not",
            r"we'd": "we would",
            r"we'll": "we will",
            r"we're": "we are",
            r"we've": "we have",
            r"weren't": "were not",
            r"what's": "what is",
            r"won't": "will not",
            r"wouldn't": "would not",
            r"you'd": "you would",
            r"you'll": "you will",
            r"you're": "you are",
            r"you've": "you have"
        }
        
        for contraction, expansion in contractions.items():
            text = re.sub(contraction, expansion, text, flags=re.IGNORECASE)
        
        return text
    
    def add_custom_stopwords(self, stopwords: List[str]):
        """
        Add custom stopwords to the cleaner
        
        Parameters:
        -----------
        stopwords : list
            List of words to add as stopwords
        """
        self.custom_stopwords.update([w.lower() for w in stopwords])
    
    def remove_custom_stopwords(self, stopwords: List[str]):
        """Remove custom stopwords"""
        for word in stopwords:
            self.custom_stopwords.discard(word.lower())
    
    def set_custom_stopwords(self, stopwords: List[str]):
        """Replace custom stopwords completely"""
        self.custom_stopwords = set([w.lower() for w in stopwords])
    
    def _get_english_stopwords(self) -> Set[str]:
        """Get common English stopwords"""
        stopwords = {
            'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am',
            'an', 'and', 'any', 'are', 'aren\'t', 'as', 'at', 'be', 'because',
            'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by',
            'can', 'can\'t', 'cannot', 'could', 'couldn\'t', 'did', 'didn\'t',
            'do', 'does', 'doesn\'t', 'doing', 'don\'t', 'down', 'during', 'each',
            'few', 'for', 'from', 'further', 'had', 'hadn\'t', 'has', 'hasn\'t',
            'have', 'haven\'t', 'having', 'he', 'he\'d', 'he\'ll', 'he\'s', 'her',
            'here', 'here\'s', 'hers', 'herself', 'him', 'himself', 'his', 'how',
            'how\'s', 'i', 'i\'d', 'i\'ll', 'i\'m', 'i\'ve', 'if', 'in', 'into',
            'is', 'isn\'t', 'it', 'it\'s', 'its', 'itself', 'just', 'k', 'let\'s',
            'm', 'me', 'me', 'might', 'more', 'most', 'mustn\'t', 'my', 'myself',
            'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other',
            'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same',
            'shan\'t', 'she', 'she\'d', 'she\'ll', 'she\'s', 'should', 'shouldn\'t',
            'so', 'some', 'such', 't', 'than', 'that', 'that\'s', 'the', 'their',
            'theirs', 'them', 'themselves', 'then', 'there', 'there\'s', 'these',
            'they', 'they\'d', 'they\'ll', 'they\'re', 'they\'ve', 'this', 'those',
            'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'wasn\'t',
            'we', 'we\'d', 'we\'ll', 'we\'re', 'we\'ve', 'were', 'weren\'t',
            'what', 'what\'s', 'when', 'when\'s', 'where', 'where\'s', 'which',
            'while', 'who', 'who\'s', 'whom', 'why', 'why\'s', 'with', 'won\'t',
            'would', 'wouldn\'t', 'y', 'you', 'you\'d', 'you\'ll', 'you\'re',
            'you\'ve', 'your', 'yours', 'yourself', 'yourselves'
        }
        return stopwords


class TextNormalizer:
    """Specialized text normalization operations"""
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize whitespace"""
        return re.sub(r'\s+', ' ', text).strip()
    
    @staticmethod
    def normalize_quotes(text: str) -> str:
        """Normalize different types of quotes to standard quotes"""
        # Replace various quote types with standard quotes
        text = re.sub(r'[\'\`\']', "'", text)  # Single quotes
        text = re.sub(r'[\"\"\„]', '"', text)  # Double quotes
        return text
    
    @staticmethod
    def normalize_hyphens(text: str) -> str:
        """Normalize different types of hyphens"""
        text = re.sub(r'[-–—]', '-', text)  # Replace all dash types with hyphen
        return text
    
    @staticmethod
    def remove_control_characters(text: str) -> str:
        """Remove control characters and zero-width characters"""
        # Remove control characters (except newline, tab, carriage return)
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' 
                      or char in '\n\t\r')
        return text
    
    @staticmethod
    def normalize_unicode(text: str, form: str = 'NFKC') -> str:
        """
        Normalize Unicode text
        
        Parameters:
        -----------
        text : str
            Text to normalize
        form : str
            Normalization form: NFC, NFKC, NFD, NFKD
        
        Returns:
        --------
        str : Normalized text
        """
        return unicodedata.normalize(form, text)
