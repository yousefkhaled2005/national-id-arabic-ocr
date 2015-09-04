from src.preprocessing.image_pipeline import DocumentImagePreprocessor
from src.decoding.beam_search import CTCBeamSearchDecoder


def test_image_preprocessor():
    info = DocumentImagePreprocessor.inspect_and_normalize(width=800, height=200)
    assert info["target_dims"][0] == 32


def test_ctc_decoder():
    decoder = CTCBeamSearchDecoder()
    text = decoder.decode_greedy([0, 1, 1, 0, 2, 0])
    assert len(text) > 0
