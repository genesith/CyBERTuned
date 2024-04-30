# Dataset wraps the data and turn it into input_ids. We perform token tagging here (for token pred task).
# DataLoader loads from Dataset. We perform masking here (for MLM) DataCollatorForLanguageModeling .

import json
import os
import pickle
import random
import time
import warnings

from dataclasses import dataclass, field

from .consts import SEMILINGUISTIC_TEXT, NONLINGUISTIC_TEXT

from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from collections.abc import Mapping

import torch
from filelock import FileLock
from torch.utils.data import Dataset
import numpy as np

from transformers.tokenization_utils import PreTrainedTokenizer, PreTrainedTokenizerBase
from transformers.utils import logging

from .iocide_mod import extract_all_modified, replace_certain_iocs

logger = logging.get_logger(__name__)


DEPRECATION_WARNING = (
    "This dataset will be removed from the library soon, preprocessing should be handled with the 🤗 Datasets "
    "library. You can have a look at this example script for pointers: {0}"
)


def find_locs_in_span(offsets,span,start_ind):
    inside = False
    indices =[]
    for ind in range(start_ind,len(offsets)):
        if not inside:
            if offsets[ind][1]>span[0]:
                inside = True
                indices.append(ind)
            else:
                continue
        if offsets[ind][0] < span[1]:
            indices.append(ind)
        else:
            return indices,ind
        



class DatasetForTokenPrediction(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        block_size: int,
        overwrite_cache=False,
        cache_dir: Optional[str] = None,
        replace_these: List = []
    ):
        warnings.warn(
            DEPRECATION_WARNING.format(
                "https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py"
            ),
            FutureWarning,
        )
        if os.path.isfile(file_path) is False:
            raise ValueError(f"Input file path {file_path} not found")

        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        possible_labels = ["None"] + SEMILINGUISTIC_TEXT + NONLINGUISTIC_TEXT
        for a in replace_these:
            possible_labels.remove(a)
        labels2id = dict()
        for i,label in enumerate(possible_labels):
            labels2id[label] = i
        self.labels2id = labels2id
        self.id2labels = possible_labels

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else directory,
            f"cached_lm_{tokenizer.__class__.__name__}_{block_size}_{filename}_{len(labels2id)}",
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.examples = []
                print("Reading file...")
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()
                
                if replace_these != []:
                    print("Replacing following iocs:", replace_these)
                    text = replace_certain_iocs(text, replace_these)

                print("Extracting spans...")
                extracted_spans = extract_all_modified(text, skip_these=replace_these)
                
                tokenized = tokenizer(text,return_offsets_mapping=True,add_special_tokens=False)
                tokenized_text = tokenized[0].ids
                token_spans = tokenized[0].offsets
                tok_labels = [0] *len(tokenized_text)

                

                for span_type in extracted_spans:
                    type_id =self.labels2id[span_type]
                    typed_spans = extracted_spans[span_type]
                    start_ind = 0
                    for _, span in typed_spans:
                        span_toks, start_ind = find_locs_in_span(token_spans,span,start_ind)
                        for span_tok in span_toks:
                            tok_labels[span_tok] = type_id


                for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                    self.examples.append({
                        'input_ids':
                        torch.tensor(tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size]),dtype=torch.long),
                        "tok_labels": torch.tensor([0]+tok_labels[i : i + block_size]+[0], dtype=torch.long),
                        }
                    )
                # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should look for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    f"Saving features into cached file {cached_features_file} [took {time.time() - start:.3f} s]"
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]
    
    def get_labels2id(self):
        return self.labels2id
    
    def get_id2labels(self):
        return self.id2labels



class DataCollatorMixin:
    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        if return_tensors == "tf":
            return self.tf_call(features)
        elif return_tensors == "pt":
            return self.torch_call(features)
        elif return_tensors == "np":
            return self.numpy_call(features)
        else:
            raise ValueError(f"Framework '{return_tensors}' not recognized!")





def _torch_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    import torch

    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.

    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result





@dataclass
class DataCollatorForTokClassandLanguageModeling(DataCollatorMixin):

    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"
    dont_mask_types: List[str] = field(default_factory=list)
    del_tok_labels: bool = False

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], batch["tok_labels"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        
        if self.del_tok_labels:
            del batch['tok_labels']
        return batch

    def torch_mask_tokens(self, inputs: Any, tok_labels: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        for skip_type in self.dont_mask_types:
            typed_mask = tok_labels == skip_type
            probability_matrix.masked_fill_(typed_mask, value=0.0)
        if probability_matrix.sum()==0:
            Exception("Nothing to mask")
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        tries = 0
        while masked_indices.sum()==0 and probability_matrix.sum()!=0:
            masked_indices = torch.bernoulli(probability_matrix).bool()
            tries+=1
            if tries %50 == 0:
                print("Stuck retrying at retry:", tries)
        if tries >0:
            print("Took many tries:", tries)
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('roberta-base')
    pog = DatasetForTokenPrediction(tok,'small_sample.txt', 512)

    pass

