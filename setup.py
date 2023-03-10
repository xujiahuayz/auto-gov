from setuptools import setup, find_packages

setup(
    name="market_env",
    packages=find_packages(),
    install_requires=[
        "numpy==1.24.2",
        "pandas==1.5.3",
        "tqdm==4.64.1",
        "torch==1.13.1",
        "gym==0.26.2",
        "opencv-python==4.7.0.68",
        "matplotlib==3.6.3",
        # "box2d-py==2.3.5",
        # "pygame==2.1.0",
        # "swig==4.1.1",
    ],
    extras_require={"dev": ["pylint", "black", "pytest"]},
)
