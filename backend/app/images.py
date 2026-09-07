"""Decoding of uploaded image bytes.

Deliberately free of torch imports so it can be exercised in the lightweight
backend test environment (pillow only) without pulling in the model stack.
"""

from __future__ import annotations

import io

from PIL import Image

from .geometry import IMG_H, IMG_W

# The model was trained on 720x480 road scenes. Anything is accepted and
# rescaled, but the geometric features use absolute pixel windows (20 px corner
# strips, 30 px edge strips), so very small inputs carry no usable signal at
# all and would otherwise get a confident-looking box back.
MIN_IMAGE_SIZE = 64


class ImageDecodeError(ValueError):
    """The uploaded bytes could not be decoded as an image."""


class ImageTooSmallError(ValueError):
    """The image decoded, but is too small for the model to say anything."""


def decode_upload(raw: bytes) -> Image.Image:
    """Decode uploaded bytes into a fully-loaded PIL image.

    Image.open() parses only the header; pixel data is decoded lazily on first
    access. A truncated or corrupt payload therefore fails later, deep inside
    the model path, where it surfaces as a 500 instead of a client error.
    Calling load() forces the decode here, while the caller can still map the
    failure onto a 400.
    """
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception as e:
        raise ImageDecodeError(str(e)) from e
    return image


def ensure_min_size(image: Image.Image, minimum: int = MIN_IMAGE_SIZE) -> None:
    """Reject images too small for the feature extractor to be meaningful."""
    if image.width < minimum or image.height < minimum:
        raise ImageTooSmallError(
            f"Image is {image.width}x{image.height}; the minimum is {minimum}x{minimum}. "
            f"The model expects {IMG_W}x{IMG_H} road scenes."
        )
