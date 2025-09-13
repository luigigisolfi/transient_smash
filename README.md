# transient_smash

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Tests status][tests-badge]][tests-link]
[![Linting status][linting-badge]][linting-link]
[![Documentation status][documentation-badge]][documentation-link]
[![License][license-badge]](./LICENSE.md)

<!-- prettier-ignore-start -->
[tests-badge]:              https://github.com/luigigisolfi/transient_smash/actions/workflows/tests.yml/badge.svg
[tests-link]:               https://github.com/luigigisolfi/transient_smash/actions/workflows/tests.yml
[linting-badge]:            https://github.com/luigigisolfi/transient_smash/actions/workflows/linting.yml/badge.svg
[linting-link]:             https://github.com/luigigisolfi/transient_smash/actions/workflows/linting.yml
[documentation-badge]:      https://github.com/luigigisolfi/transient_smash/actions/workflows/docs.yml/badge.svg
[documentation-link]:       https://github.com/luigigisolfi/transient_smash/actions/workflows/docs.yml
[license-badge]:            https://img.shields.io/badge/License-BSD_3--Clause-blue.svg
<!-- prettier-ignore-end -->

`transient_smash` is a Python package for detecting the presence of transient astrophysical events in time-series data.
It uses a Bayesian model comparison approach, computing the evidence ratio (Bayes factor) for generative models for the time-series data,
using simulated based inference algorithms.

This project was developed as part of the [ASTRODAT25 workshop](https://bronreichardtchu.github.io/ASTRODAT/).

## About

### Project Team

- Emma Godden ([emmagodden123](https://github.com/emmagodden123))
- Jack Davey ([JJD333](https://github.com/JJD333))
- Giulia Borghetto ([gborghetto](https://github.com/gborghetto))
- Luigi Gisolfi ([luigigisolfi](https://github.com/luigigisolfi))
- Matt Graham ([matt-graham](https://github.com/matt-graham))

## Getting Started

### Prerequisites

<!-- Any tools or versions of languages needed to run code. For example specific Python or Node versions. Minimum hardware requirements also go here. -->

`transient_smash` requires Python 3.11&ndash;3.13.

### Installation

<!-- How to build or install the application. -->

We recommend installing in a project specific virtual environment created using
a environment management tool such as
[Conda](https://docs.conda.io/projects/conda/en/stable/). To install the latest
development version of `transient_smash` using `pip` in the currently active
environment run

```sh
pip install git+https://github.com/luigigisolfi/transient_smash.git
```

Alternatively create a local clone of the repository with

```sh
git clone https://github.com/luigigisolfi/transient_smash.git
```

and then install in editable mode by running

```sh
pip install -e .
```

### Running Locally

How to run the application on your local system.

### Running Tests

<!-- How to run tests on your local system. -->

Tests can be run across all compatible Python versions in isolated environments
using [`tox`](https://tox.wiki/en/latest/) by running

```sh
tox
```

To run tests manually in a Python environment with `pytest` installed run

```sh
pytest tests
```

again from the root of the repository.

### Building Documentation

The MkDocs HTML documentation can be built locally by running

```sh
tox -e docs
```

from the root of the repository. The built documentation will be written to
`site`.

Alternatively to build and preview the documentation locally, in a Python
environment with the optional `docs` dependencies installed, run

```sh
mkdocs serve
```

## Roadmap

- [x] Initial Research
- [ ] Minimum viable product <-- You are Here
- [ ] Alpha Release
- [ ] Feature-Complete Release
