Contribution Guidelines
=================================

Contributions are welcomed and can be made to the public Git repository available at: https://github.com/McMasterRS/LFSpy

We encourage anyone looking to contribute to consult the open issues available at https://github.com/McMasterRS/LFSpy/issues

We ask that in submitting changes you consult the coding standards and pull request guidelines outlined below.

Contributing to the method:
---------------------------------
This library impliments the Localized Feature Selection method outlined by Nargus Armenford. As such, changes made the method should be only done to reflect changes made to the theoretical basis.

Coding Standards:
---------------------------------

In general the code should be written according to the PEP 8 style standard for code and documentation. A guide to PEP8 can be found at https://www.python.org/dev/peps/pep-0008/

Submitting a Pull Request
---------------------------------

Please submit one pull request per feature. Before submitting a pull request ensure your code continues to pass the unit tests. LFSpy uses pytest and the tests are located in the tests directory of this repository.

The tests can be run using the command::

        pytest --pyargs LFSpy