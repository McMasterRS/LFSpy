import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="LFSpy",
    version="1.0.1",
    author="Oliver Cook, Kiret Dhindsa, Areeb Khawaja, Ron Harwood, Thomas Mudway",
    install_requires=['numpy>=1.14', 'scipy>=1.1', 'scikit-learn>=0.18.2', 'pytest>=5.0.0'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/McMasterRS/LFS/",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)   