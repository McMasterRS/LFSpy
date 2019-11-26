import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="LFSpy",
    version="1.0.0",
    author="Narges Armanfard, Oliver Cook, Kiret Dhindsa, Areeb Khawaja, Ron Harwood, Thomas Mudway",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/McMasterRS/LFS/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)