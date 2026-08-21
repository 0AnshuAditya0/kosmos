from datasets import load_dataset

ds = load_dataset("ai4bharat/MSMARCO-XI", streaming=True, split="train")

print("Dataset loaded, now pulling first row...")
row = next(iter(ds))
print("Got a row:")
print(row)