from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="calyx",
    version="0.1.0",
    author="CALYX Team",
    description="Statically Typed Prompts for Python - Write prompts like functions. Enforce structure like Pydantic.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Maykon-fernanado/CALYX",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "PyYAML>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=5.0",
        ],
        "llm": [
            "openai>=1.0.0",
            "anthropic>=0.3.0",
        ],
        "all": [
            "pytest>=7.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=5.0",
            "openai>=1.0.0",
            "anthropic>=0.3.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "CALYX=CALYX:main",
        ],
    },
    keywords="ai llm prompts typing validation yaml json anthropic openai pydantic",
    project_urls={
        "Bug Reports": "https://github.com/Maykon-fernanado/CALYX/issues",
        "Source": "https://github.com/Maykon-fernanado/CALYX",
        "Documentation: "https://github.com/Maykon-fernanado/CALYX/tree/main",
    },
    package_data={
        "calyx": ["schemas/*.yaml", "examples/*.yaml", "examples/*.json"],
    },
    include_package_data=True,
)
