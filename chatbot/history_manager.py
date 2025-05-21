import os
import json

HISTORY_FILE = "conversation_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def add_message(history, role, content):
    history.append({"role": role, "content": content})
    save_history(history)
