import logging

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from df_engine.core.keywords import RESPONSE, PRE_TRANSITIONS_PROCESSING, GLOBAL, TRANSITIONS, LOCAL
from df_engine.core import Actor
from df_engine import conditions as cnd

from df_transitions.annotators import HFProximityAnnotator, HFClassifierAnnotator
from df_transitions.types import IntentCollection
from df_transitions import conditions as i_cnd

from examples import example_utils

logger = logging.getLogger(__name__)

tokenizer = AutoTokenizer("bert-base-uncased")
model = AutoModelForSequenceClassification("my-classification-model")

annotator_1 = HFClassifierAnnotator(
    tokenizer=tokenizer, model=model, intent_collection=IntentCollection.parse_yaml("./data/example.yaml")
)
annotator_2 = HFProximityAnnotator(
    tokenizer=tokenizer, model=model, intent_collection=IntentCollection.parse_yaml("./data/example.yaml")
)


script = {
    GLOBAL: {
        PRE_TRANSITIONS_PROCESSING: {"get_intents_1": annotator_1, "get_intents_2": annotator_2},
        TRANSITIONS: {("food", "offer", 2): i_cnd.user_has_intent("food")},
    },
    "root": {
        LOCAL: {TRANSITIONS: {("service", "offer", 1.2): cnd.true()}},
        "start": {RESPONSE: "Hi!"},
        "fallback": {RESPONSE: "I can't quite get what you mean."},
        "finish": {RESPONSE: "Ok, see you soon!", TRANSITIONS: {("root", "start", 1.3): cnd.true()}},
    },
    "service": {"offer": {RESPONSE: "What would you like me to look up?"}},
    "food": {
        "offer": {
            RESPONSE: "Would you like me to look up a restaurant for you?",
            TRANSITIONS: {
                ("food", "no_results", 1.2): cnd.regexp(r"yes|yeah|good|ok|yep"),
                ("root", "finish", 0.8): cnd.true(),
            },
        },
        "no_results": {
            RESPONSE: "Sorry, all the restaurants are closed due to COVID restrictions.",
            TRANSITIONS: {("root", "finish"): cnd.true()},
        },
    },
}

actor = Actor(script, start_label=("root", "start"), fallback_label=("root", "fallback"))


def main():
    logging.basicConfig(
        format="%(asctime)s-%(name)15s:%(lineno)3s:%(funcName)20s():%(levelname)s - %(message)s",
        level=logging.INFO,
    )
    example_utils.run_interactive_mode(actor)


if __name__ == "__main__":
    main()
