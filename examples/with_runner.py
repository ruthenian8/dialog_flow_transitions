from df_runner import ScriptRunner

from examples.regexp import regex_scorer, script

runner = ScriptRunner(
    script,
    start_label=("root", "start"),
    fallback_label=("root", "fallback"),
    pre_annotators=[regex_scorer],
    post_annotators=[],
)

if __name__ == "__main__":
    runner.start()
