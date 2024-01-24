# Contributing to Pearl
We want to make contributing to this project as easy and transparent as
possible.

## Our Development Process
New models and new modules developed by external parties will be carefully reviewed
by the Applied Reinforcement Learning team at Meta to make sure it is indeed useful
to the broader community and also up to our development standards. Please start an
issue before starting any pull requests.

## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes (see [How to run tests](#how-to-run-tests) for specifics).
5. Make sure your code lints.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

## How to run tests
You can run the test suite with the following command ran from the project root directory (the `Pearl` directory containing `pearl` and `test` subdirectories):
```
python -m unittest discover -t . <package to be tested>
```

The `test` package contains `unit` and `integration` subpackages. To run all tests, run:
```
python -m unittest discover -t . test
```

To run unit tests, run:
```
python -m unittest discover -t . test.unit
```

To run integration tests, run:
```
python -m unittest discover -t . test.integration
```

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Meta's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Meta has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## Coding Style
* Please follow code style presented in our repo. We will strictly enforcing
code style standards for contributions.

## License
By contributing to Pearl, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
