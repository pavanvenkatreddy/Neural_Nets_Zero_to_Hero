from BPE import Tokenizer

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Create an instance of the Tokenizer class
tokenizer = Tokenizer(300)

tokenizer.train(text)

print(tokenizer.encode("hello world"))
print(tokenizer.decode(tokenizer.encode("hello world")))