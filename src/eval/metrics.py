"""
Character Error Rate (CER) and Word Error Rate (WER) Evaluation Engine
Copyright (c) 2014-2017 Yousef Khaled.
"""


def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def compute_cer(reference: str, hypothesis: str) -> float:
    ref_len = len(reference)
    if ref_len == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    dist = levenshtein_distance(reference, hypothesis)
    return dist / ref_len


def compute_wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    dist = levenshtein_distance(" ".join(ref_words), " ".join(hyp_words))
    return dist / len(ref_words)
