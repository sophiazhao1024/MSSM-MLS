# Contributing

This repository preserves a historical-method reconstruction and a documented
failure. Please keep changes traceable to evidence.

- Run `python -m pytest -q` in a dedicated Python 3.12 environment.
- Separate software fixes, changes justified by newly found historical sources,
  and modern algorithm experiments. A change to training, transforms, cuts,
  stopping, point selection or physics settings is a scientific change.
- Do not label tuned settings as recovered historical settings. Record the
  exact source and distinguish the paper, xBIT, and reconstruction choices.
- Never count predicted points as true solutions, replace tool errors with
  physical negative labels, or silently fill failed recommendations with random
  points. Keep target policy and budget definitions explicit.
- Generated runs, machine configurations, third-party installations and private
  paths must stay untracked. The included saved-data fixture is an explicitly
  sanitized derivative, not a cache-import journal.
- `RELEASE_MANIFEST.json` is the integrity record of this particular source
  release. Its verification is expected to fail after intentional edits. For a
  new release, review those edits and update version, changelog, test evidence
  and manifest together. Do not regenerate a manifest merely to conceal a
  mismatch in downloaded reference files.
- CI tests Python contracts and frozen evidence only. A green CI result is not
  a validation of the external physics installation or a reproduced efficiency.

Contributions to the reconstruction are distributed under GPL-3.0-or-later;
retain upstream notices and identify imported third-party code separately.

