from setuptools import setup, find_packages


setup(
    name="plf_env",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "ethereum-tools==0.1.5",
        "python-dotenv",
        "requests",
        "stringcase",
        "tqdm",
        "matplotlib",
    ],
    extras_require={"dev": ["pylint", "black", "pytest"]},
    entry_points={
        "console_scripts": ["plf-env=plf_env.cli:run"],
    },
)
