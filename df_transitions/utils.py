STATUS_UNAVAILABLE = 503
STATUS_SUCCESS = 200

INTENT_KEY = "intents"


class UnknownIntentError(Exception):
    pass


def singleton(cls: type):
    def singleton_inner(*args, **kwargs):
        if singleton_inner.instance is None:
            singleton_inner.instance = cls(*args, **kwargs)
        return singleton_inner.instance

    singleton_inner.instance = None
    return singleton_inner
