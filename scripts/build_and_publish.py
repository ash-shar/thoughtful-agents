#!/usr/bin/env python
"""
Build and publish the package to PyPI.

Usage:
    python scripts/build_and_publish.py [--test]

Options:
    --test  Publish to TestPyPI instead of PyPI
"""

import os
import sys
import subprocess
import shutil

def run_command(command):
    """Run a shell command and print its output."""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, check=True)
    return result

def main():
    # Parse arguments
    test_mode = "--test" in sys.argv

    # Clean up previous builds
    print("Cleaning up previous builds...")
    if os.path.exists("dist"):
        shutil.rmtree("dist")
    if os.path.exists("build"):
        shutil.rmtree("build")
    if os.path.exists("thoughtful_agents.egg-info"):
        shutil.rmtree("thoughtful_agents.egg-info")

    # Install build dependencies
    print("Installing build dependencies...")
    run_command("pip install --upgrade pip setuptools wheel twine")

    # Build the package
    print("Building the package...")
    run_command("python setup.py sdist bdist_wheel")

    # Check the package
    print("Checking the package...")
    run_command("twine check dist/*")

    # Upload to PyPI or TestPyPI
    if test_mode:
        print("Uploading to TestPyPI...")
        run_command("twine upload --repository testpypi dist/*")
        print("\nPackage published to TestPyPI!")
        print("You can install it with:")
        print("pip install --index-url https://test.pypi.org/simple/ thoughtful-agents")
    else:
        print("Uploading to PyPI...")
        run_command("twine upload dist/*")
        print("\nPackage published to PyPI!")
        print("You can install it with:")
        print("pip install thoughtful-agents")

if __name__ == "__main__":
    main() 