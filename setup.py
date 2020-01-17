import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="LFSpy",
    version="1.0.2-dev",
    author="Oliver Cook, Kiret Dhindsa, Areeb Khawaja, Ron Harwood, Thomas Mudway",
    install_requires=['numpy>=1.14', 'scipy>=1.1', 'scikit-learn>=0.18.2'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/McMasterRS/LFS/",
    packages=setuptools.find_packages(),
    package_data={
        'LFSpy': ['tests/matlab_Data.mat'],
        },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    tests_require=['pytest>=5.0.0'],
    python_requires='>=3.6',
    extras_require={
        'test': ['pytest>=5.0.0']
        }
)
