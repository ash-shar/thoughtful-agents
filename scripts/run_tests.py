#!/usr/bin/env python
"""
Run tests for the thoughtful-agents package.

Usage:
    python scripts/run_tests.py
"""

import unittest
import sys
import os

def main():
    """Run all tests in the tests directory."""
    # Add the parent directory to the path so we can import the package
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    # Discover and run tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Return non-zero exit code if tests failed
    sys.exit(not result.wasSuccessful())

if __name__ == "__main__":
    main() 