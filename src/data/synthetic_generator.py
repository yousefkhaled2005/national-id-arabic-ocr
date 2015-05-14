"""
Synthetic Arabic Text Line Generator for CRNN Pretraining
Copyright (c) 2014-2017 Yousef Khaled.
"""

import numpy as np


class SyntheticDataGenerator:
    """
    Renders Arabic text lines with synthetic noise, blurring, and skew angles to pretrain CRNN feature extractors.
    """
    def __init__(self, img_height: int = 32, img_width: int = 256):
        self.img_height = img_height
        self.img_width = img_width

    def generate_dummy_sample(self, text: str) -> np.ndarray:
        # Returns normalized float32 tensor simulating optical scan noise
        canvas = np.ones((self.img_height, self.img_width), dtype=np.float32) * 0.95
        noise = np.random.normal(0, 0.05, canvas.shape).astype(np.float32)
        return np.clip(canvas + noise, 0.0, 1.0)
