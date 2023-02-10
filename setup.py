from setuptools import setup, find_packages

setup(
    name="market_env",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "tqdm",
    ],
    extras_require={"dev": ["pylint", "black", "pytest"]},
)
