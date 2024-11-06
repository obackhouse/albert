"""Configuration file for `pytest`."""

import hashlib
import inspect

import numpy as np
import pytest


class Helper:
    """Helper class for tests."""

    @staticmethod
    def random(shape):
        """Generate a deterministic array that appears random.

        Each call to this function will return a different array, but the array will always be
        the same between runs for a given call (as long as the code is not modified).
        """
        caller = inspect.currentframe().f_back
        location = ":".join(
            [
                caller.f_code.co_filename.split("/")[-1],
                caller.f_code.co_name,
                str(caller.f_lineno),
            ]
        )
        iden = int(hashlib.sha256(location.encode()).hexdigest(), 16) % int(1e10)
        size = np.prod(shape)
        array = np.cos(np.arange(size) + iden).reshape(shape)
        return array

    @staticmethod
    def fingerprint(array):
        """Find the fingerprint of an array."""
        return np.cos(np.arange(array.size)) @ array.ravel()


@pytest.fixture
def helper():
    """Fixture for the helper class."""
    return Helper()
