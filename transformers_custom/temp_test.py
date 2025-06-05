from datasets import load_dataset

# Load a small sample of the dataset
dataset = load_dataset("opus_books", "en-it", split='train')

# Print the structure of the dataset
print(dataset)

# Print a few examples
for example in dataset:
    print(example)
    break  # Remove this line if you want to see more examples