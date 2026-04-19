# src/utils.py
import time
import numpy as np
import random
import os

def set_global_seed(seed: int = 42):
    """Fixe la seed pour numpy, random et (plus tard) TF/PyTorch."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

class Timer:
    def __init__(self):
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start_time
