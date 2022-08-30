"""
Training
*********

This module contains utilities for HuggingFace classifier fine-tuning.
"""
try:
    IMPORT_ERROR_MESSAGE = None
    from torch.utils.data import Dataset
    from tokenizers import Tokenizer
except ImportError as e:
    IMPORT_ERROR_MESSAGE = e.msg
    Dataset = object
    Tokenizer = object

from .types import Label, LabelCollection


class TrainingDataset(Dataset):
    def __init__(self, tokenizer: Tokenizer, label_collection: LabelCollection) -> None:
        IMPORT_ERROR_MESSAGE = globals().get("IMPORT_ERROR_MESSAGE")
        if IMPORT_ERROR_MESSAGE is not None:
            raise ImportError(IMPORT_ERROR_MESSAGE)
        super().__init__()
        self._data = []
        for label in label_collection.labels.values():
            label: Label
            for sentence in label.examples:
                self._data += [(tokenizer(sentence), label._categorical_code)]

    def __getitem__(self, idx):
        return self._data[idx]

    def __len__(self):
        return len(self._data)
