"""
Run all tests
"""
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    # Run all tests in tests directory
    pytest.main([
        "tests/",
        "-v",
        "--tb=short"
    ])
