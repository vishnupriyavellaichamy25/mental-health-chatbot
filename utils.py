# ─────────────────────────────────────────────────────────────────────────────
# utils.py — Shared utility functions for Minds on Fire
# ─────────────────────────────────────────────────────────────────────────────

import os
import logging
import time
import pickle
from pathlib import Path
from typing import Any

import numpy as np

# ── Logging setup ─────────────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a consistently formatted logger for any module.

    Args:
        name:  Usually __name__ of the calling module.
        level: Logging verbosity (default INFO).

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:                       # avoid duplicate handlers
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ── Path helpers ──────────────────────────────────────────────────────────────

def project_root() -> Path:
    """Return the absolute path of the project root (directory of this file)."""
    return Path(__file__).parent.resolve()


def data_dir() -> Path:
    """Ensure data/ exists and return its path."""
    d = project_root() / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def models_dir() -> Path:
    """Ensure models/ exists and return its path."""
    m = project_root() / "models"
    m.mkdir(parents=True, exist_ok=True)
    return m


# ── Serialisation helpers ─────────────────────────────────────────────────────

def save_pickle(obj: Any, path: str | Path) -> None:
    """
    Persist a Python object to disk with pickle.

    Args:
        obj:  Any pickle-able object.
        path: Destination file path.
    """
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str | Path) -> Any:
    """
    Load a pickle file from disk.

    Args:
        path: File path to load.

    Returns:
        Deserialised Python object.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pickle not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Basic text sanitation: strip whitespace, collapse newlines, remove nulls.

    Args:
        text: Raw input string.

    Returns:
        Cleaned string.
    """
    if not isinstance(text, str):
        return ""
    text = text.strip()
    # Collapse excessive whitespace and newlines
    text = " ".join(text.split())
    return text


# ── Timing helpers ────────────────────────────────────────────────────────────

class Timer:
    """
    Context-manager / manual timer for latency measurements.

    Usage:
        with Timer() as t:
            do_something()
        print(t.elapsed_ms)   # milliseconds
    """

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self._end = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return (self._end - self._start) * 1_000


# ── Cosine similarity helper ──────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two 1-D vectors.

    Args:
        a, b: Float numpy arrays of the same shape.

    Returns:
        Scalar similarity in [-1, 1].
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ── Crisis resource strings ───────────────────────────────────────────────────

CRISIS_RESOURCES = """
---
🆘 **You're not alone. Free & confidential support is available right now:**
- **iCall (India):** 📞 9152987821 *(Mon–Sat, 8 am–10 pm)*
- **Vandrevala Foundation:** 📞 1860-2662-345 *(24/7, all languages)*
- **iCall Chat:** https://icallhelpline.org/

*If you are in immediate danger, please call emergency services (112).*
---
"""

DISTRESS_RESOURCES = """
---
💙 **It sounds like things are tough right now. You can reach out for support:**
- **iCall (India):** 📞 9152987821
- **Vandrevala Foundation:** 📞 1860-2662-345 *(24/7)*
---
"""

DISCLAIMER = (
    "⚠️ **Disclaimer:** Minds on Fire is an AI companion for psychoeducation and "
    "peer-style support. It is **not** a substitute for professional mental health "
    "care. Always consult a licensed therapist or psychiatrist for clinical concerns."
)
