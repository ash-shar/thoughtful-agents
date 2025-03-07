from setuptools import setup, find_packages

setup(
    name="inner-thoughts-ai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "openai>=1.0.0",
        "spacy>=3.0.0",
        "typing-extensions>=4.0.0",  # For better typing support
    ],
    author="",
    author_email="",
    description="A framework for modeling agent thoughts and conversations",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/liubruce/inner-thoughts-ai",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache 2.0 License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
) 