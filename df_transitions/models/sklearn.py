from typing import Optional, Union
import joblib

from ..types import LabelCollection

try:
    from sklearn.base import BaseEstimator
    from sklearn.pipeline import make_pipeline, Pipeline
    from scipy.sparse import csr_matrix
except ImportError as e:
    BaseEstimator = object
    Pipeline = object
    make_pipeline = object
    csr_matrix = object
    IMPORT_ERROR_MESSAGE = e.msg

from .base_model import BaseModel


class BaseSklearnModel(BaseModel):
    """
    Base class for Sklearn-based models.

    Parameters
    ----------
    model: Optional[BaseEstimator]
        Sklearn Estimator for prediction. Should implement 'fit' and 'predict' methods.
        Example: LogisticRegression.
    tokenizer: Optional[Union[BaseEstimator, Pipeline]]
        Sklearn Estimator for preprocessing. Should implement 'fit' and 'transform' methods.
        Example: TfidfVectorizer. Can also be a product of several preprocessors,
        unified with a pipeline.
    namespace_key: Optional[str]
        Name of the namespace in framework states that the model will be using.
    """

    def __init__(
        self,
        model: Optional[BaseEstimator] = None,
        tokenizer: Optional[Union[BaseEstimator, Pipeline]] = None,
        namespace_key: Optional[str] = None,
    ) -> None:
        IMPORT_ERROR_MESSAGE = globals().get("IMPORT_ERROR_MESSAGE")
        if IMPORT_ERROR_MESSAGE is not None:
            raise ImportError(IMPORT_ERROR_MESSAGE)
        assert tokenizer is not None, "tokenizer parameter is required."
        super().__init__(namespace_key=namespace_key)
        self.model = model
        self.tokenizer = tokenizer
        self._pipeline = make_pipeline(*[tokenizer] + ([model] if model else []))

    def transform(self, request: str):
        intermediate_result = self._pipeline.transform([request])
        if isinstance(intermediate_result, csr_matrix):
            return intermediate_result.toarray()
        return intermediate_result

    def fit(self, label_collection: LabelCollection):
        sentences = [sentence for label in label_collection.labels.values() for sentence in label.examples]
        pred_labels = [
            number
            for label in label_collection.labels.values()
            for number in [label._categorical_code] * len(label.examples)
        ]
        self._pipeline.fit(sentences, pred_labels)

    def save(self, path: str, **kwargs) -> None:
        joblib.dump(self.model, f"{path}.model")
        joblib.dump(self.tokenizer, f"{path}.tokenizer")

    @classmethod
    def load(cls, path: str, namespace_key: str) -> __qualname__:
        model = joblib.load(f"{path}.model")
        tokenizer = joblib.load(f"{path}.tokenizer")
        return cls(model=model, tokenizer=tokenizer, namespace_key=namespace_key)
