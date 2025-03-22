from BPE import Tokenizer

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Create an instance of the Tokenizer class
tokenizer = Tokenizer(500)

#Train it on the text
tokenizer.train(text)

# Save the tokenizer for later use
tokenizer.save('tokenizer.pth')