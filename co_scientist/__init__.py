"""AIDO Co-Scientist: automated ML model building for biological datasets."""

import os as _os

# ── Prevent OpenMP segfaults ─────────────────────────────────────────────────
# Multiple native libraries (torch, sklearn, xgboost, lightgbm) ship their own
# copy of libomp.dylib. Loading them together causes segfaults on macOS.
# Set these BEFORE any native library is imported.
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# ─────────────────────────────────────────────────────────────────────────────

__version__ = "0.1.0"
