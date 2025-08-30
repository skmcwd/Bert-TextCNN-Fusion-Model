from datasets import load_dataset

dataset = load_dataset(path='datasets', split='train')
print(len(dataset))
print(dataset[0:1000])