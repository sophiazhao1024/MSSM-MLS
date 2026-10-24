# Changelog

## 0.1.1 — source release preparation

- Package the raw-input, log-omega-only historical-method reconstruction for a public repository.
- Keep the numerical learning, sampling, physical adapters, drivers and default scan settings byte-identical to the inspected 0.1.0 variant.
- Add English and Chinese documentation, explicit reconstruction assumptions, citations, GPL-3.0-or-later licensing for the reconstruction, software CI, and a release manifest.
- Publish a sanitized derivative of the initial 1000 saved truth records, preserving every learning input and label; remove local execution paths and command metadata.
- Include the frozen first-fit weights and an exact arithmetic check of their failure to reach the target windows.
- Update distribution metadata and `__version__` only; adapt diagnostic paths to the repository layout and use restricted weight loading.

Known result: the reference saved-data first round exhausts 20,000,000 proposals with zero recommendations. This release does not fix that failure, recover unpublished author settings, or demonstrate the paper's approximately 80% efficiency.

