from typing import List, Dict


class CRNNModelConfig:
    """Architecture configuration for 7-layer CNN + 2-layer BiLSTM + CTC."""

    def __init__(self):
        self.conv_filters = [64, 128, 256, 256, 512, 512, 512]
        self.rnn_hidden_units = 256
        self.vocab = [
            "blank", "ا", "ب", "ت", "ث", "ج", "ح", "خ", "د", "ذ", "ر", "ز", "س", "ش",
            "ص", "ض", "ط", "ظ", "ع", "غ", "ف", "ق", "ك", "ل", "م", "ن", "ه", "و", "ي",
            "ء", "ئ", "ؤ", "ة", "ى", "٠", "١", "٢", "٣", "٤", "٥", "٦", "٧", "٨", "٩"
        ]

    def get_num_classes(self) -> int:
        return len(self.vocab)
