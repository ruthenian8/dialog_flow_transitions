"""
Training
---------

This module contains utilities for HuggingFace classifier fine-tuning.
"""
try:
    from torch.utils.data import Dataset
except ImportError:
    Dataset = object

from tokenizers import Tokenizer

from .types import Intent, IntentCollection


class IntentDataset(Dataset):
    def __init__(self, tokenizer: Tokenizer, intent_collection: IntentCollection) -> None:
        super().__init__()
        self._data = []
        for intent in intent_collection.items.values():
            intent: Intent
            for sentence in intent.positive_examples:
                self._data += [(tokenizer(sentence), intent._categorical_code)]

    def __getitem__(self, idx):
        return self._data[idx]

    def __len__(self):
        return len(self._data)
