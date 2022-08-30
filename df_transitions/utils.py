import re

STATUS_UNAVAILABLE = 503
STATUS_SUCCESS = 200

LABEL_KEY = "labels"


class DefaultTokenizer:
    def __call__(self, string: str):
        return re.split(r"\W+", string=string)
