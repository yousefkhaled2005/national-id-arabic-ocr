"""
Arabic Character Vocabulary and Glyph Encoding for CRNN-CTC
Copyright (c) 2014-2017 Yousef Khaled.
"""

ARABIC_CHARACTERS = [
    " ", "ا", "أ", "إ", "آ", "ب", "ت", "ث", "ج", "ح", "خ", "د", "ذ", "ر", "ز", "س",
    "ش", "ص", "ض", "ط", "ظ", "ع", "غ", "ف", "ق", "ك", "ل", "م", "ن", "ه", "و", "ي",
    "ى", "ة", "ء", "ئ", "ؤ", "٠", "١", "٢", "٣", "٤", "٥", "٦", "٧", "٨", "٩"
]

CHAR_TO_INDEX = {c: i + 1 for i, c in enumerate(ARABIC_CHARACTERS)}
INDEX_TO_CHAR = {i + 1: c for i, c in enumerate(ARABIC_CHARACTERS)}
BLANK_INDEX = 0

def encode_label(text: str):
    return [CHAR_TO_INDEX[c] for c in text if c in CHAR_TO_INDEX]

def decode_indices(indices):
    return "".join([INDEX_TO_CHAR[i] for i in indices if i in INDEX_TO_CHAR])
