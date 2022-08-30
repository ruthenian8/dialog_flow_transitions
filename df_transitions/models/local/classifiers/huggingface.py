"""
HuggingFace classifier model
------------------------------

TODO: complete
"""
from typing import List

from ...huggingface import BaseHFModel


class HFClassifier(BaseHFModel):
    """
    HFClassifier utilizes Hugging Face models to predict utterance labels.

    Parameters
    -----------
    model: PreTrainedModel
        A pretrained Hugging Face format model.
    tokenizer: Tokenizer
        A pretrained Hugging Face tokenizer.
    device: torch.device
        Pytorch device object. The device will be used for inference and pre-training.
    namespace_key: str
        Name of the namespace in framework states that the model will be using.
    tokenizer_kwargs: Optional[dict] = None
        Default tokenizer arguments override.
    model_kwargs: Optional[dict] = None
        Default model arguments override.
    """

    def predict(self, request: str) -> dict:
        model_output = self.call_model(request)
        logits_list = model_output.logits
        predicted_class_id = logits_list.argmax().item()
        label = self.model.config.id2label[predicted_class_id]
        result = {label: 1} # TODO: probs ?
        return result
