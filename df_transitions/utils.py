import re

STATUS_UNAVAILABLE = 503
STATUS_SUCCESS = 200

INTENT_KEY = "intents"


class DefaultTokenizer:
    def __call__(self, string: str):
        return re.split(r"\W+", string=string)
