'''
Takes raw text and saves ALBERT-cased features for that text to disk

Adapted from BERT conversion script using Huggingface Transformers

'''
import torch
from transformers import AlbertTokenizerFast
from transformers import AutoModel
from argparse import ArgumentParser
import h5py
import numpy as np

argp = ArgumentParser()
argp.add_argument('--input_path', type=str, required=True)
argp.add_argument('--output_path', type=str, required=True)
argp.add_argument('--model', help='e.g., albert-base-v2', type=str, required=True)
argp.add_argument('--tokenizer', help='e.g., albert-base-v2', type=str, required=True)
args = argp.parse_args()

# Load pre-trained model tokenizer (vocabulary)
# Crucially, do not do basic tokenization; PTB is tokenized. Just do wordpiece tokenization.

tokenizer = AlbertTokenizerFast.from_pretrained(args.tokenizer)
model = AutoModel.from_pretrained(args.model)
LAYER_COUNT = 12
FEATURE_COUNT = 768

model.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

with h5py.File(args.output_path, 'w') as fout:
  for index, line in enumerate(open(args.input_path)):
    line = line.strip() # Remove trailing characters
    line = '[CLS] ' + line + ' [SEP]'
    tokenized_text = tokenizer.tokenize(line)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segment_ids = [1 for x in tokenized_text]

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens]).to(device)
    segments_tensors = torch.tensor([segment_ids]).to(device)

    with torch.no_grad():
        result = model(input_ids=tokens_tensor, token_type_ids=segments_tensors, output_hidden_states=True)
        encoded_layers = result[2][1:]
    dset = fout.create_dataset(str(index), (LAYER_COUNT, len(tokenized_text), FEATURE_COUNT))
    dset[:,:,:] = np.vstack([np.array(x.detach().cpu()) for x in encoded_layers])
  


