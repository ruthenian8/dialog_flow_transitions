"""
Regex Model
----------------

This module provides a regex-based label model version.
"""
import re
from typing import Optional, Union

from ...base_model import BaseModel
from ....types import LabelCollection


class RegexModel:
    """
    RegexModel implements utterance classification based on regex rules.

    Parameters
    -----------
    label_collection: LabelCollection
        Labels for the matcher. The prediction output depends on proximity to different labels.
    """

    def __init__(self, label_collection: LabelCollection):
        self.label_collection = label_collection

    def __call__(self, request: str, **re_kwargs):
        result = {}
        for label_name, label in self.label_collection.labels.items():
            matches = [re.search(item, request, **re_kwargs) for item in label.examples]
            if any(map(bool, matches)):
                result[label_name] = 1.0

        return result


class RegexClassifier(BaseModel):
    """
    RegexClassifier wraps a :py:class:`~RegexModel` for label annotation.

    Parameters
    -----------
    model: RegexModel
        An instance of df_transitions' RegexModel.
    namespace_key: Optional[str]
        Name of the namespace in framework states that the model will be using.
    re_kwargs: Optional[dict]
        re arguments override.
    """

    def __init__(
        self,
        model: Union[RegexModel, LabelCollection],
        namespace_key: str,
        re_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__(namespace_key=namespace_key)
        self.re_kwargs = re_kwargs or {"flags": re.IGNORECASE}
        # instantiate if Label Collection has been passed
        self.model = model if isinstance(model, RegexModel) else RegexModel(model)

    def fit(self, label_collection: LabelCollection):
        self.model.label_collection = label_collection

    def predict(self, request: str) -> dict:
        return self.model(request, **self.re_kwargs)
