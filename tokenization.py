import re
from typing import List, Dict
from neuroflex import NeuralTokenizer

class CustomTokenizer(NeuralTokenizer):
    def __init__(self, vocab_size: int = 50000, max_token_length: int = 16):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_token_length = max_token_length
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.token_pattern = re.compile(r'\w+|[^\w\s]')

    def fit(self, texts: List[str]):
        word_freq = {}
        for text in texts:
            words = self.token_pattern.findall(text.lower())
            for word in words:
                if word not in word_freq:
                    word_freq[word] = 0
                word_freq[word] += 1

        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        for i, (word, _) in enumerate(sorted_words[:self.vocab_size - 2]):
            self.vocab[word] = i + 2
            self.inverse_vocab[i + 2] = word

        self.vocab['<PAD>'] = 0
        self.vocab['<UNK>'] = 1
        self.inverse_vocab[0] = '<PAD>'
        self.inverse_vocab[1] = '<UNK>'

    def encode(self, text: str) -> List[int]:
        words = self.token_pattern.findall(text.lower())
        return [self.vocab.get(word, 1) for word in words]  # 1 is <UNK> token

    def decode(self, tokens: List[int]) -> str:
        return ' '.join([self.inverse_vocab.get(token, '<UNK>') for token in tokens])

    def tokenize(self, text: str) -> List[str]:
        return self.token_pattern.findall(text.lower())

# Example usage
if __name__ == "__main__":
    texts = [
        "Hello, world! This is a test.",
        "NeuroFlex is an advanced neural network framework.",
        "Custom tokenization for language models."
    ]

    tokenizer = CustomTokenizer(vocab_size=100)
    tokenizer.fit(texts)

    sample_text = "Hello, NeuroFlex! This is a custom tokenizer."
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)

    print(f"Original: {sample_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
