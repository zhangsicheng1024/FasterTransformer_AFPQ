from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

out_file = "/workspace/mpt-ft/FasterTransformerModified/build/out"

with open(out_file, "r") as fp:
    line = fp.readline()
    ids = line.strip().split(" ")
    out_str = tokenizer.decode([int(id) for id in ids])
    print(out_str)
