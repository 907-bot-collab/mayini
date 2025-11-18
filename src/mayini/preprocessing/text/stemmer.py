
from typing import List, Dict, Optional
import re


class PorterStemmer:
    """
    Complete Porter Stemmer implementation
    Reduces words to their root stem using morphological rules
    Original algorithm: M. Porter, 1980
    """
    
    def __init__(self):
        """Initialize stemmer with step tables"""
        self._initialize_step2_list()
        self._initialize_step3_list()
    
    def stem(self, word: str) -> str:
        """
        Stem a single word
        
        Parameters:
        -----------
        word : str
            Word to stem
        
        Returns:
        --------
        str : Stemmed word
        """
        word = word.lower()
        
        # Words shorter than 3 characters are returned unchanged
        if len(word) <= 2:
            return word
        
        # Apply 5-step stemming process
        word = self._step1a(word)
        word = self._step1b(word)
        word = self._step1c(word)
        word = self._step2(word)
        word = self._step3(word)
        word = self._step4(word)
        word = self._step5(word)
        
        return word
    
    def stem_list(self, words: List[str]) -> List[str]:
        """
        Stem a list of words
        
        Parameters:
        -----------
        words : list
            List of words to stem
        
        Returns:
        --------
        list : List of stemmed words
        """
        return [self.stem(word) for word in words]
    
    # ===== HELPER METHODS =====
    
    def _measure(self, word: str) -> int:
        """
        Calculate the measure of a word
        Measure = number of VC patterns in the word
        V = vowel, C = consonant
        """
        cv_sequence = ''
        vowels = 'aeiou'
        
        # Build CV pattern
        for char in word:
            if char in vowels:
                cv_sequence += 'V'
            else:
                cv_sequence += 'C'
        
        # Count VC patterns
        measure = 0
        for i in range(len(cv_sequence) - 1):
            if cv_sequence[i] == 'C' and cv_sequence[i+1] == 'V':
                measure += 1
        
        return measure
    
    def _contains_vowel(self, word: str) -> bool:
        """Check if word contains a vowel"""
        vowels = 'aeiou'
        return any(char in vowels for char in word)
    
    def _double_consonant(self, word: str) -> bool:
        """Check if word ends with a double consonant"""
        if len(word) < 2:
            return False
        
        vowels = 'aeiouywx'
        return (word[-1] == word[-2] and word[-1] not in vowels)
    
    def _cvc(self, word: str) -> bool:
        """
        Check if word ends with consonant-vowel-consonant
        where the last consonant is not w, x, or y
        """
        if len(word) < 3:
            return False
        
        vowels = 'aeiou'
        consonants_to_exclude = 'wxy'
        
        return (word[-3] not in vowels and
                word[-2] in vowels and
                word[-1] not in vowels and
                word[-1] not in consonants_to_exclude)
    
    def _initialize_step2_list(self):
        """Initialize mapping for step 2"""
        self.step2_list = {
            'ational': 'ate', 'tional': 'tion', 'enci': 'ence',
            'anci': 'ance', 'izer': 'ize', 'bli': 'ble',
            'alli': 'al', 'entli': 'ent', 'eli': 'e',
            'ousli': 'ous', 'ization': 'ize', 'ation': 'ate',
            'ator': 'ate', 'alism': 'al', 'iveness': 'ive',
            'fulness': 'ful', 'ousness': 'ous', 'aliti': 'al',
            'iviti': 'ive', 'biliti': 'ble', 'logi': 'log'
        }
    
    def _initialize_step3_list(self):
        """Initialize mapping for step 3"""
        self.step3_list = {
            'icate': 'ic', 'ative': '', 'alize': 'al',
            'iciti': 'ic', 'ical': 'ic', 'ful': '',
            'ness': ''
        }
    
    # ===== STEMMING STEPS =====
    
    def _step1a(self, word: str) -> str:
        """
        Step 1a: Plurals
        SSES -> SS, IES -> I, SS -> SS, S -> (nothing)
        """
        if word.endswith('sses'):
            return word[:-2]
        elif word.endswith('ies'):
            return word[:-3] + 'i'
        elif word.endswith('ss'):
            return word
        elif word.endswith('s'):
            return word[:-1]
        
        return word
    
    def _step1b(self, word: str) -> str:
        """
        Step 1b: Past tense
        EED -> EE (if stem contains vowel)
        ED/ING -> (null) (if stem contains vowel)
        """
        if word.endswith('eed'):
            stem = word[:-3]
            if self._measure(stem) > 0:
                return word[:-1]
            else:
                return word
        
        elif word.endswith('ed') or word.endswith('ing'):
            stem = word[:-2] if word.endswith('ed') else word[:-3]
            
            if self._contains_vowel(stem):
                word = stem
                
                # Additional rules
                if word.endswith('at') or word.endswith('bl') or word.endswith('iz'):
                    return word + 'e'
                elif len(word) >= 2 and word[-1] == word[-2] and word[-1] not in 'lsz':
                    return word[:-1]
                elif self._measure(word) == 1 and self._cvc(word):
                    return word + 'e'
        
        return word
    
    def _step1c(self, word: str) -> str:
        """
        Step 1c: Replace Y/y
        (*v*) Y -> I (e.g., happy -> happi)
        """
        if len(word) > 1 and word[-1] in 'yY':
            if word[-2] not in 'aeiou':
                return word[:-1] + 'i'
        
        return word
    
    def _step2(self, word: str) -> str:
        """
        Step 2: Mapping
        Applies mappings listed in step2_list
        """
        for suffix, replacement in self.step2_list.items():
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if self._measure(stem) > 0:
                    return stem + replacement
        
        return word
    
    def _step3(self, word: str) -> str:
        """
        Step 3: Mapping
        Applies mappings listed in step3_list
        """
        for suffix, replacement in self.step3_list.items():
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if self._measure(stem) > 0:
                    return stem + replacement
        
        return word
    
    def _step4(self, word: str) -> str:
        """
        Step 4: Remove a final -e if m > 1, and other conditions
        """
        suffixes_m_greater_1 = ['al', 'ance', 'ence', 'er', 'ic', 'able', 
                               'ible', 'ant', 'ement', 'ment', 'ent', 'ou']
        
        for suffix in suffixes_m_greater_1:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if self._measure(stem) > 1:
                    return stem
        
        # Special case for -ion
        if word.endswith('ion') and len(word) >= 4:
            if word[-4] in 'st':
                stem = word[:-3]
                if self._measure(stem) > 1:
                    return stem
        
        return word
    
    def _step5(self, word: str) -> str:
        """
        Step 5a: Remove a final -e if m > 1, or m = 1 and not *d and not -L
        Step 5b: Remove final -ll if m > 1 and *d L
        """
        # Step 5a
        if word.endswith('e'):
            stem = word[:-1]
            measure = self._measure(stem)
            
            if measure > 1:
                return stem
            elif measure == 1:
                if not self._cvc(stem):
                    return stem
        
        # Step 5b
        if self._measure(word) > 1 and self._double_consonant(word) and word.endswith('ll'):
            return word[:-1]
        
        return word


class SimpleStemmer:
    """
    Simple suffix-based stemmer for basic stemming needs
    Faster than Porter but less accurate
    """
    
    def __init__(self):
        """Initialize with common suffix rules"""
        self.suffix_rules = [
            ('ies', 'i'),
            ('es', 'e'),
            ('ed', ''),
            ('ing', ''),
            ('ings', ''),
            ('ly', ''),
            ('ness', ''),
            ('ment', ''),
            ('ful', ''),
            ('less', ''),
            ('able', ''),
            ('ible', ''),
            ('tion', ''),
            ('sion', ''),
            ('er', ''),
            ('est', ''),
        ]
    
    def stem(self, word: str, min_stem_length: int = 3) -> str:
        """
        Apply simple suffix-based stemming
        
        Parameters:
        -----------
        word : str
            Word to stem
        min_stem_length : int
            Minimum length for stem
        
        Returns:
        --------
        str : Stemmed word
        """
        word = word.lower()
        
        if len(word) < 4:
            return word
        
        for suffix, replacement in self.suffix_rules:
            if word.endswith(suffix):
                stem = word[:-len(suffix)] + replacement
                
                if len(stem) >= min_stem_length:
                    return stem
        
        return word
    
    def stem_list(self, words: List[str]) -> List[str]:
        """Stem a list of words"""
        return [self.stem(word) for word in words]


class LancasterStemmer:
    """
    Lancaster Stemmer - Another stemming algorithm
    More aggressive than Porter Stemmer
    """
    
    def __init__(self):
        """Initialize Lancaster stemmer"""
        self.step2_list = self._get_step2_rules()
    
    def stem(self, word: str) -> str:
        """
        Stem word using Lancaster algorithm
        """
        word = word.lower()
        
        if len(word) <= 2:
            return word
        
        return self._strip_prefixes(word)
    
    def _strip_prefixes(self, word: str) -> str:
        """Strip common prefixes and suffixes"""
        prefixes = ['kilo', 'mega', 'giga', 'anti', 'multi', 'pseudo']
        
        for prefix in prefixes:
            if word.startswith(prefix):
                return word[len(prefix):]
        
        # Apply suffix rules
        for suffix, replacement in self.step2_list.items():
            if word.endswith(suffix):
                return word[:-len(suffix)] + replacement
        
        return word
    
    def _get_step2_rules(self) -> Dict[str, str]:
        """Get step 2 rules for Lancaster stemmer"""
        return {
            'ational': 'ate', 'tional': 'tion', 'enci': 'ence',
            'anci': 'ance', 'izer': 'ize', 'bli': 'ble',
            'alli': 'al', 'entli': 'ent', 'eli': 'e',
            'ousli': 'ous', 'ization': 'ize', 'ation': 'ate'
        }
    
    def stem_list(self, words: List[str]) -> List[str]:
        """Stem a list of words"""
        return [self.stem(word) for word in words]
