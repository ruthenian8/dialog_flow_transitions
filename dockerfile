FROM rasa/rasa:3.2.6-full
RUN rasa init --no-prompt
ENTRYPOINT rasa run --enable-api -m models/$(ls /app/models)
