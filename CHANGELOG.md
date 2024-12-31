# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Add cpu and gpu extra dependencies in pyproject.toml, users can install torch with trajdl[cpu] or trajdl[gpu]
- Add cibuildwheel in dev dependencies
- Add submodules in trajdl_cpp, including grid and distance
  - Add functions to compute discrete frechet distance

### Changed
- Move some functions in trajdl_cpp into grid submodule

## [0.1.0] - 2024-12-21

### Added
- Implemented single and multi-sequence data structures for trajectory processing
- Added support for Gowalla and Porto datasets with automatic download and caching
- Integrated unified LightningModule for model training
- Implemented Tokenizer and GridSystem for Location sequence and trajectory discretization
- Added deep learning algorithms:
  - TULER
  - t2vec
  - GM-VSAE
  - ST-LTSM
- Created initial documentation for v0.1.0

### Changed
- None

### Deprecated
- None

### Removed
- None

### Fixed
- None

### Security
- None

[0.1.0]: https://github.com/Spatial-Temporal-Data-Mining/TrajDL/releases/tag/v0.1.0
