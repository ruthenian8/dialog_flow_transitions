from df_runner import ScriptRunner

from examples.regexp import regex_model, script

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
