from setuptools import setup, find_packages

setup(
    name="jailbreak-harness",
    version="0.2.1",
    author="Casey Fahey",
    license="MIT",  
    description="LLM jailbreak testing harness for security research",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/CaseyFaheyNetgoalie/jailbreak_harness.py",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Security",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "openai>=1.3.0",
        "requests>=2.31.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "pandas>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.10.0",
            "flake8>=6.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "jailbreak-harness=jailbreak_harness.harness:main",
        ],
    },
)
