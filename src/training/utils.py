# src/training/utils.py
#
# A centralised logging, JSON saving/loading, and keyword heuristics source file. Keep other scripts stay lean and reusable.

import json
import logging
import os
from datetime import datetime


# -----------------------------
# Logging Setup
# -----------------------------
def setup_logger(name="judge_logger", log_file="training.log", level=logging.INFO):
    """Set up a logger with console + file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


# -----------------------------
# JSON Helpers
# -----------------------------
def save_json(data, filepath):
    """Save Python dict/list to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return filepath


def load_json(filepath):
    """Load JSON file into Python object."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# Keyword Heuristic (Baseline)
# -----------------------------
def heuristic_is_good(agent_output):
    """
    Simple baseline heuristic:
    Predict 'good' if agent output contains scheduling intent keywords.
    """
    keywords = ["schedule", "invite", "calendar", "meeting", "call"]
    text = agent_output.lower()
    return any(k in text for k in keywords)


# -----------------------------
# Timestamp Helper
# -----------------------------
def timestamp():
    """Return current timestamp string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
