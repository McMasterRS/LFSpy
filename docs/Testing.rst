Testing
=================================

We recommend running the provided test after installing LFSpy to ensure the results obtained match expected outputs.

pytest may be installed either directly through pip (pip install pytest) or using the test extra (pip install LFSpy[test]).::

    pytest --pyargs LFSpy

This will output to console whether or not the results of LFSpy on two datasets (the sample dataset provided in this repository, and scikit-learn's Fisher Iris dataset) are exactly as expected.

So far, LFSpy has been tested on Windows 10 with and without Conda, and on Ubuntu. In all cases, results have been exactly the expected results.