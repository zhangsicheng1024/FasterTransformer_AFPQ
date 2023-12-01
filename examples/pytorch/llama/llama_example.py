import torch
import torch.distributed as dist
from transformers import AutoTokenizer


lib_path = "/workspace/mpt-ft/FasterTransformerModified/build/lib/libth_transformer.so"
llama_2_config = "/workspace/mpt-ft/FasterTransformerModified/examples/cpp/llama/llama_2_config.ini"

# Need even only one process
try:
    dist.init_process_group(backend='mpi')
except:
    print("[INFO] WARNING: Have initialized the process group")

torch.classes.load_library(lib_path)

model = torch.classes.FasterTransformer.LlamaOp(llama_2_config)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"

sentences = ["Hey, are you consciours? Can you talk to me?"]
bs = len(sentences)

encoded_inputs = tokenizer.batch_encode_plus(sentences)

# Must be padded to multiple of 16, otherwise will get wrong result
input_ids = torch.IntTensor(tokenizer.pad(encoded_inputs, pad_to_multiple_of=16)["input_ids"]).cuda()
input_lengths = torch.IntTensor([len(ids) for ids in input_ids]).cuda()

bad_words = torch.IntTensor([[7768, 3908], [1, 2]]).cuda()
stop_words = torch.IntTensor([[[287, 4346, 12], [3, -1, -1]]] * bs).cuda()
output_ids, sequence_lengths, output_log_probs = model.forward(
    input_ids,     # input_ids
    input_lengths, # input_lengths
    100,           # output_len
    bad_words,     # bad_words_list_opt <optional>,
    stop_words,    # stop_words_list_opt <optional>,
)

print(tokenizer.decode(output_ids[0][0][:sequence_lengths[0][0]]))
