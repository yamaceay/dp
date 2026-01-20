from typing import List, Any
from tokenizers import Encoding
import itertools
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast
import torch

device = (
    'cuda' if torch.cuda.is_available() else 
    'mps' if torch.backends.mps.is_available() else
    'cpu'
)

IntList = List[int]
IntListList = List[IntList]

class LabelSet:
    def __init__(self, labels: List[str]):
        self.labels_to_id = {}
        self.ids_to_label = {}
        self.labels_to_id["O"] = 0
        self.ids_to_label[0] = "O"
        num = 0 
        for _num, (label, s) in enumerate(itertools.product(labels, "BI")):
            num = _num + 1
            l = f"{s}-{label}"
            self.labels_to_id[l] = num
            self.ids_to_label[num] = l

    def get_aligned_label_ids_from_annotations(self, tokenized_text, annotations, ids):
        raw_labels, identifier_types, offsets, ids = align_tokens_and_annotations_bilou(tokenized_text, annotations, ids)
        return list(map(self.labels_to_id.get, raw_labels)), identifier_types, offsets, ids

@dataclass
class TrainingExample:
    input_ids: IntList
    attention_masks: IntList
    labels: IntList
    identifier_types: IntList
    offsets:IntList

class Dataset(Dataset):
    def __init__(
        self,
        data: Any,
        label_set: LabelSet,
        tokenizer: PreTrainedTokenizerFast,
        tokens_per_batch=32,
        window_stride=None,
    ):
        self.label_set = label_set
        if window_stride is None:
            self.window_stride = tokens_per_batch
        self.tokenizer = tokenizer
    
        self.texts = []
        self.annotations = []
        ids = []

        for example in data:
            self.texts.append(example["text"])
            self.annotations.append(example["annotations"])
            ids.append(example['doc_id'])

        tokenized_batch = self.tokenizer(self.texts, add_special_tokens=True, return_offsets_mapping=True)

        offset_mapping = []
        for x,y in zip(ids, tokenized_batch.offset_mapping):
            l = []
            for tpl in y:
                l.append((x, tpl[0], tpl[1]))
            offset_mapping.append(l)

        aligned_labels = []
        identifiers = []
        o = []
        for ix in range(len(tokenized_batch.encodings)):
            encoding = tokenized_batch.encodings[ix]
            raw_annotations = self.annotations[ix]
            aligned, identifier_types, outs, ids= label_set.get_aligned_label_ids_from_annotations(
                encoding, raw_annotations, ids
            )
            aligned_labels.append(aligned)
            identifiers.append(identifier_types)
            o.append(outs)

        self.training_examples: List[TrainingExample] = []
        for encoding, label, identifier_type, mapping  in zip(tokenized_batch.encodings, aligned_labels, identifiers, offset_mapping):
            length = len(label)
            for start in range(0, length, self.window_stride):
                end = min(start + tokens_per_batch, length)
                padding_to_add = 0
                self.training_examples.append(
                    TrainingExample(
                        input_ids=encoding.ids[start:end]
                        + [self.tokenizer.pad_token_id]
                        * padding_to_add,
                        labels=(
                            label[start:end]
                            + [-1] * padding_to_add
                        ),
                        attention_masks=(
                            encoding.attention_mask[start:end]
                            + [0]
                            * padding_to_add
                        ),
                        identifier_types=(identifier_type[start:end]
                            + [-1] * padding_to_add
                        
                        ),
                        offsets=(mapping[start:end]
                            + [-1] * padding_to_add
                        ),

                    )
                )

    def __len__(self):
        return len(self.training_examples)

    def __getitem__(self, idx) -> TrainingExample:

        return self.training_examples[idx]

class TrainingBatch:
    def __getitem__(self, item):
        return getattr(self, item)

    def __init__(self, examples: List[TrainingExample]):
        self.input_ids: torch.Tensor
        self.attention_masks: torch.Tensor
        self.labels: torch.Tensor
        self.identifier_types: List
        self.offsets:List
        input_ids: IntListList = []
        masks: IntListList = []
        labels: IntListList = []
        identifier_types: List = []
        offsets: List = []

        for ex in examples:
            input_ids.append(ex.input_ids)
            masks.append(ex.attention_masks)
            labels.append(ex.labels)
            identifier_types.append(ex.identifier_types)
            offsets.append(ex.offsets)
        self.input_ids = torch.LongTensor(input_ids)
        self.attention_masks = torch.LongTensor(masks)
        self.labels = torch.LongTensor(labels)
        self.identifier_types = identifier_types
        self.offsets = offsets

        self.input_ids = self.input_ids.to(device)
        self.attention_masks = self.attention_masks.to(device)
        self.labels = self.labels.to(device)

def align_tokens_and_annotations_bilou(tokenized: Encoding, annotations, ids):
    tokens = tokenized.tokens
    identifier_types = ["O"] * len(
        tokens
    )
    aligned_labels = ["O"] * len(
        tokens
    )
    offsets = ["O"] * len(
        tokens
    )
    for anno in annotations:
        ids.append(anno['id'])
        if anno['label'] == 'MASK':
            annotation_token_ix_set = (
                set()
            )
            for char_ix in range(anno["start_offset"], anno["end_offset"]):

                token_ix = tokenized.char_to_token(char_ix)
                if token_ix is not None:
                    annotation_token_ix_set.add(token_ix)
            for num, token_ix in enumerate(sorted(annotation_token_ix_set)):
                if num == 0:
                    prefix = "B"
                else:
                    prefix = "I"
                aligned_labels[token_ix] = f"{prefix}-{anno['label']}"
                identifier_types[token_ix] = anno['identifier_type']
                offsets[token_ix] = {anno['id'] : (anno['start_offset'], anno['end_offset'])}

    return aligned_labels, identifier_types, offsets, ids
