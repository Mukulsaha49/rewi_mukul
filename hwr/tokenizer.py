import json
from typing import List, Dict


class Tokenizer:
    def __init__(self, vocab_path: str):
        """
        Loads token vocabulary from a JSON file.
        Expects <BLANK> and <UNK> as part of the vocab.
        """
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.token_to_index: Dict[str, int] = json.load(f)
        
        self.index_to_token: Dict[int, str] = {
            int(v): k for k, v in self.token_to_index.items()
        }

        # Important attributes
        self.vocab = list(self.token_to_index.keys())
        self.vocab_size = len(self.token_to_index)
        self.blank_token = "<BLANK>"
        self.unk_token = "<UNK>"

        self.blank_index = self.token_to_index[self.blank_token]
        self.unk_index = self.token_to_index[self.unk_token]

    def tokenize(self, text: str, n: int = 2) -> List[str]:
        """
        Tokenize input string into n-grams (default bigrams).
        """
        text = text.strip()
        return [text[i:i+n] for i in range(len(text) - n + 1)] if len(text) >= n else []

    def encode(self, text: str, n: int = 2) -> List[int]:
        """
        Convert text into list of token indices.
        Unknown tokens are mapped to <UNK>.
        """
        tokens = self.tokenize(text, n)
        return [self.token_to_index.get(tok, self.unk_index) for tok in tokens]

    def decode(self, indices: List[int], remove_duplicates: bool = False, ignore_blank: bool = True) -> str:
        """
        Convert token indices back to string using overlapping bigrams.
        Designed for CTC decoding.
        """
        tokens = []
        prev_id = None

        for idx in indices:
            if ignore_blank and idx == self.blank_index:
                continue
            if remove_duplicates and idx == prev_id:
                continue
            token = self.index_to_token.get(idx, self.unk_token)
            tokens.append(token)
            prev_id = idx

        # Reconstruct from overlapping bigrams
        if not tokens:
            return ""
        output = tokens[0]
        for token in tokens[1:]:
            output += token[-1]  # Add only last char of bigram
        return output

    def save(self, path: str):
        """
        Save token_to_index mapping to JSON.
        """
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.token_to_index, f, ensure_ascii=False, indent=4)

    def get_vocab(self) -> List[str]:
        """
        Return list of all tokens in vocabulary.
        """
        return self.vocab
