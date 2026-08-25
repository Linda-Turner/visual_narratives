import base64

import torch


def encode_image(image_path):
    """
    Encode an image file as a base64-encoded UTF-8 string.

    Args:
        image_path: Path to the image file.

    Returns:
        str: Base64-encoded representation of the image.
    """
    with open(image_path, "rb") as image_file:

        return base64.b64encode(image_file.read()).decode('utf-8')


def get_device():
    """
    Determine the device available for PyTorch computation.

    Returns:
        str: "cuda" if a CUDA-enabled GPU is available, otherwise "cpu".
    """
    return "cuda" if torch.cuda.is_available() else "cpu"