from df_runner import ScriptRunner

from examples.regexp import annotator, script

runner = ScriptRunner(
    script,
    start_label=("root", "start"),
    fallback_label=("root", "fallback"),
    pre_annotators=[annotator],
    post_annotators=[],
)

if __name__ == "__main__":
    runner.start()
