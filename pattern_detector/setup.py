from setuptools import setup, find_packages

setup(
    name="pattern-detector",
    version="0.1.0",
    description="A library for pattern detection in signals.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author=["Ahmet Faruk Minareci", "Yigit Utku Bulut"],
    author_email=["ahmetfaruk.minareci@gmail.com","yigit.utku.bulut@outlook.com"],
    url="https://github.com/bulutyigit/pattern_detector",  # Replace with your GitHub repo
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)