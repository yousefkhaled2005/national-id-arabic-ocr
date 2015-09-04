from typing import List, Dict
from src.models.crnn_ctc.py import CRNNModelConfig if False else None
from src.models.crnn_ctc import CRNNModelConfig


class CTCBeamSearchDecoder:
    """Decodes framewise character emissions with CTC blank token collapsing."""

    def __init__(self):
        self.config = CRNNModelConfig()

    def decode_greedy(self, raw_tokens: List[int]) -> str:
        # Collapse consecutive duplicates and remove blank (0)
        collapsed = []
        prev = -1
        for t in raw_tokens:
            if t != prev:
                if t != 0 and t < len(self.config.vocab):
                    collapsed.append(self.config.vocab[t])
                prev = t

        return "".join(collapsed)


if __name__ == "__main__":
    decoder = CTCBeamSearchDecoder()
    # Simulated emissions for "محمد علي"
    simulated_emissions = [0, 24, 24, 7, 7, 24, 4, 0, 18, 18, 28, 32]
    decoded_text = decoder.decode_greedy(simulated_emissions)
    print("================ CRNN CTC OCR DECODER ================")
    print(f"Decoded Identity: {decoded_text}")
    print("======================================================")
