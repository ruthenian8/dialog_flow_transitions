from df_runner import ScriptRunner

from examples.base_example import regex_model, script

# TODO: waiting for new version of runner
runner = ScriptRunner(
    script,
    start_label=("root", "start"),
    fallback_label=("root", "fallback"),
    pre_annotators=[regex_model],
    post_annotators=[],
)

if __name__ == "__main__":
    runner.start()
