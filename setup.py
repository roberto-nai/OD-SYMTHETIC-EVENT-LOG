from setuptools import setup


setup(
    name="synthetic-log-generator",
    version="0.1.0",
    py_modules=["log_to_synth"],
    python_requires=">=3.10,<3.11",
    install_requires=[
        "numpy",
        "pandas",
        "torch",
    ],
)
