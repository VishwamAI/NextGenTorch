from setuptools import setup, find_packages

setup(
    name="NextGenTorch",
    version="0.1.0",
    author="VishwamAI",
    author_email="contact@vishwamai.com",
    description="NextGenTorch is a powerful and flexible PyTorch-based library for next-generation language models.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/VishwamAI/NextGenTorch",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.18.0",
        "datasets>=2.0.0",
        "ollama>=0.1.0",
        "neuroflex>=0.1.0",
        "fairscale>=0.4.5",
        "langchain>=0.0.139",
        "pytest>=6.2.5",
        "jax>=0.4.1",
        "flax>=0.6.0",
        "sentencepiece>=0.1.96",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "nextgentorch-chat=NextGenTorch.chat_interface:main",
        ],
    },
)
