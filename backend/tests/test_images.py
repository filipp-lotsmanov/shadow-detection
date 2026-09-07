"""Upload decoding contract.

These only import app.images (pillow, no torch), so they run in the
lightweight backend environment alongside the schema tests.

The case that matters is the truncated upload: Image.open() succeeds on a
partial file because it reads only the header, so without an explicit load()
the failure lands inside the model path and the API answers 500 instead of
400.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from app.images import (
    MIN_IMAGE_SIZE,
    ImageDecodeError,
    ImageTooSmallError,
    decode_upload,
    ensure_min_size,
)


def _png_bytes(size=(720, 480), mode="RGB", color=(120, 120, 120)) -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_valid_png_decodes():
    image = decode_upload(_png_bytes())
    assert image.size == (720, 480)


def test_rgba_png_decodes():
    """The bundled sample images are RGBA; the feature extractor drops the
    alpha channel later, so decoding must not reject them."""
    image = decode_upload(_png_bytes(mode="RGBA", color=(120, 120, 120, 255)))
    assert image.mode == "RGBA"


def test_truncated_png_raises_decode_error():
    raw = _png_bytes()
    with pytest.raises(ImageDecodeError):
        decode_upload(raw[: len(raw) // 3])


def test_non_image_bytes_raise_decode_error():
    with pytest.raises(ImageDecodeError):
        decode_upload(b"# Not an image, just markdown\n")


def test_empty_bytes_raise_decode_error():
    with pytest.raises(ImageDecodeError):
        decode_upload(b"")


def test_decode_error_is_a_value_error():
    """main.py maps ImageDecodeError onto HTTP 400. Keeping it a ValueError
    subclass means a caller that only knows about ValueError still behaves."""
    assert issubclass(ImageDecodeError, ValueError)


def test_native_size_passes_the_minimum():
    ensure_min_size(decode_upload(_png_bytes()))


def test_square_at_the_minimum_passes():
    ensure_min_size(decode_upload(_png_bytes(size=(MIN_IMAGE_SIZE, MIN_IMAGE_SIZE))))


@pytest.mark.parametrize("size", [(1, 1), (8, 8), (63, 200), (200, 63)])
def test_undersized_images_are_rejected(size):
    """A tiny upload used to come back with a confident-looking bbox; the
    geometric features use absolute pixel windows, so there is no signal."""
    image = decode_upload(_png_bytes(size=size))
    with pytest.raises(ImageTooSmallError):
        ensure_min_size(image)


def test_too_small_error_names_both_dimensions():
    image = decode_upload(_png_bytes(size=(10, 12)))
    with pytest.raises(ImageTooSmallError, match="10x12"):
        ensure_min_size(image)
