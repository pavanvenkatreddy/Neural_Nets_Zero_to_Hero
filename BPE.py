import torch
import regex as re

class Tokenizer:
    def __init__(self, size=None, model_path=None):
        """
        At initialization: size or model_path
        When to train: text to train(text)
        """
        if model_path:
            # Load saved vocab and merges from a single file
            saved_model = torch.load(model_path)
            self.vocab = saved_model['vocab']
            self.merges = saved_model['merges']
            self.size = len(self.vocab)
        else:
            self.size = size
    
    def stats(self, ids, counts=None):
        """Function to generate the stats"""
        counts = {} if counts is None else counts
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def merge(self, ids, pair, idx):
        """Function to merge the tokens across the dataset"""
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
                # when the merge happens it jumps 2
            else:
                newids.append(ids[i])
                i += 1
                # when it doesn't it jumps 1
        return newids

    def train(self, text):
        """Train the tokenizer on the provided text."""
        # regex pattern to split
        # Pattern is taken from the GPT-4 Tokenizer
        pattern = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        re_pattern = re.compile(pattern)
        text_chunks = re_pattern.findall(text)
        
        # building basic vocab of all possible characters
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        # generating tokens for the text chunks
        tokens = [list(ch.encode("utf-8")) for ch in text_chunks]
        # Dictionary to maintain the performed merges
        self.merges = {}
        for i in range(256, self.size):
            # stats to get the most commonly occurring sequential pair
            seq_stats = {}
            for chunk_ids in tokens:
                # generates all the stats for the given chunks and appends them to the stats
                self.stats(chunk_ids, seq_stats)
            # fetches the top pair
            top_pair = max(seq_stats, key=seq_stats.get)
            # performs the merge
            tokens = [self.merge(chunk_ids, top_pair, i) for chunk_ids in tokens]
            # updates the vocab based on the performed merge
            self.vocab[i] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]
            # save in the merge dict
            self.merges[top_pair] = i
    
    def decode(self, ids):
        """Function to decode the given tokens"""
        tokens = b"".join([self.vocab[id] for id in ids])
        text = tokens.decode("utf-8", errors="replace")
        return text
    
    def encode(self, text):
        """Function to encode the given text"""
        # first encode the given text to utf-8
        tokens = list(text.encode("utf-8"))
        # loops over to replace the tokens if the no. of tokens are greater than 1
        while len(tokens) >= 2:
            numerics = self.stats(tokens)
            # the min below tries to find the pairs in self.merges and selects the one with least index i.e: ((111, 112), 270) over ((112, 113), 271)
            pair = min(numerics, key=lambda p: self.merges.get(p, float("inf")))
            # when all the merges are completed it breaks the loop
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens
        
    def save(self, model_path):
        """Save vocab and merges in a single file"""
        model = {'vocab': self.vocab, 'merges': self.merges}
        torch.save(model, model_path)
        print(f"Saved model to {model_path}")