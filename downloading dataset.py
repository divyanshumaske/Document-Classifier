from datasets import load_dataset

dataset = load_dataset(
    "aharley/rvl_cdip",
    cache_dir="C:/Users/divya/OneDrive/Documents/Document classifier/data"
)

print(dataset)