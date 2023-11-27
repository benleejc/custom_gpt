import os

class AppConfig:

    def __init__(self):
        self.OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
