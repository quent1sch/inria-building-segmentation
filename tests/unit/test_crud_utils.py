"""
tests/unit/test_crud_utils.py

Unit tests for pure utility functions in db/crud.py.

Tested (no DB connection needed):
  - hash_bytes: SHA256 of raw bytes
  - compute_input_hash: determinism, param ordering, canonical ref sensitivity

Not tested here (require async DB session):
  - create_job, get_job, mark_done, etc. (integration territory)
"""

import hashlib
import json

import pytest

from db.crud import compute_input_hash, hash_bytes


# ── hash_bytes ────────────────────────────────────────────────────────────────

class TestHashBytes:

    def test_known_value(self):
        """SHA256 of b"hello" is a known constant."""
        expected = hashlib.sha256(b"hello").hexdigest()
        assert hash_bytes(b"hello") == expected

    def test_returns_64_char_hex_string(self):
        """SHA256 digest is always 64 hex characters."""
        result = hash_bytes(b"some data")
        assert isinstance(result, str)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_empty_bytes(self):
        """SHA256 of empty bytes is a known constant."""
        expected = hashlib.sha256(b"").hexdigest()
        assert hash_bytes(b"") == expected

    def test_different_inputs_different_hashes(self):
        assert hash_bytes(b"abc") != hash_bytes(b"def")

    def test_same_input_same_hash(self):
        """Deterministic — same input always produces same hash."""
        assert hash_bytes(b"test") == hash_bytes(b"test")


# ── compute_input_hash ────────────────────────────────────────────────────────

class TestComputeInputHash:

    def test_returns_64_char_hex(self):
        result = compute_input_hash("some/path.tif", {"resolution": 0.1})
        assert isinstance(result, str)
        assert len(result) == 64

    def test_deterministic(self):
        """Same inputs → same hash every time."""
        params = {"resolution": 0.1, "processing": "clean", "result_type": "mask"}
        h1 = compute_input_hash("/data/tile.tif", params)
        h2 = compute_input_hash("/data/tile.tif", params)
        assert h1 == h2

    def test_different_refs_different_hashes(self):
        """Different file paths → different hashes."""
        params = {"resolution": 0.1}
        h1 = compute_input_hash("/data/tile_a.tif", params)
        h2 = compute_input_hash("/data/tile_b.tif", params)
        assert h1 != h2

    def test_different_params_different_hashes(self):
        """Same file, different params → different hash (cache miss expected)."""
        ref = "/data/tile.tif"
        h1 = compute_input_hash(ref, {"resolution": 0.1, "processing": "raw"})
        h2 = compute_input_hash(ref, {"resolution": 0.1, "processing": "clean"})
        assert h1 != h2

    def test_param_order_does_not_matter(self):
        """
        Params dict is serialised with sort_keys=True so insertion order
        should not affect the hash.
        """
        ref    = "/data/tile.tif"
        params_a = {"resolution": 0.1, "processing": "clean", "result_type": "mask"}
        params_b = {"result_type": "mask", "processing": "clean", "resolution": 0.1}
        assert compute_input_hash(ref, params_a) == compute_input_hash(ref, params_b)

    def test_none_resolution_included_in_hash(self):
        """
        None values are serialised as null in JSON, so they affect the hash.
        resolution=None vs resolution=0.1 → different hashes.
        """
        ref = "/data/tile.tif"
        h1 = compute_input_hash(ref, {"resolution": None})
        h2 = compute_input_hash(ref, {"resolution": 0.1})
        assert h1 != h2

    def test_empty_params_valid(self):
        """Empty params dict should not crash."""
        result = compute_input_hash("/data/tile.tif", {})
        assert len(result) == 64

    def test_url_ref_works(self):
        """URLs as canonical_ref should work the same as file paths."""
        url = "https://example.blob.core.windows.net/container/tile.tif?sas=xyz"
        result = compute_input_hash(url, {"resolution": 0.1})
        assert len(result) == 64

    def test_separator_prevents_collision(self):
        """
        Without a separator between ref and params, "ab" + "c" would equal
        "a" + "bc". The || separator prevents this class of collision.
        """
        h1 = compute_input_hash("ab", {"c": 1})
        h2 = compute_input_hash("a",  {"bc": 1})
        # These should differ because the payload strings differ
        # "ab||{"c": 1}" vs "a||{"bc": 1}"
        assert h1 != h2