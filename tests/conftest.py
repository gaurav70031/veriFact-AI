"""
Pytest configuration for the test suite.

Sets asyncio_mode = "auto" so all async test functions run automatically
without needing @pytest.mark.asyncio on every method.
"""
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )
