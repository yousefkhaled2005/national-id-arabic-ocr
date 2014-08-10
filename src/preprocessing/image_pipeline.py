from typing import Dict, Tuple


class DocumentImagePreprocessor:
    """Prepares scanned ID card images for CRNN sequence extraction."""

    TARGET_HEIGHT = 32
    TARGET_WIDTH = 256

    @staticmethod
    def inspect_and_normalize(width: int, height: int) -> Dict[str, any]:
        aspect_ratio = width / max(height, 1)
        scale_factor = DocumentImagePreprocessor.TARGET_HEIGHT / max(height, 1)
        new_width = int(width * scale_factor)

        return {
            "original_dims": (width, height),
            "aspect_ratio": round(aspect_ratio, 2),
            "target_dims": (DocumentImagePreprocessor.TARGET_HEIGHT, min(new_width, DocumentImagePreprocessor.TARGET_WIDTH)),
            "binarization_method": "otsu_adaptive"
        }
