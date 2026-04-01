#!/usr/bin/env python3
# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
from h2o_sonar import version


def get_new_release_changelog_section() -> str:
    sem_ver = version.__version__
    section = f"""
## [v{sem_ver}](https://github.com/h2oai/h2o-sonar/tree/v{sem_ver}) — 2026-??-??

This is a minor H2O Sonar release.

### Added

* **Evaluators**:
    * .
* **Features**:
    * .
* **Enhancements**:
    * .
* **Documentation**
    * .

### Fixed

No fixes.

### Changed

No changes.

### Deprecated

No deprecations.

### Removed

No removals.

### Security

No security fixes.
"""
    # print(section)
    return section


def add_new_release_section_to_changelog():
    section = get_new_release_changelog_section()
    anchor = "and this project adheres to [Semantic Versioning](http://semver.org/)."
    section_to_inject = f"{anchor}\n\n\n{section}"

    with open("CHANGELOG.md", "r") as f:
        changelog = f.read()

    new_changelog=changelog.replace(anchor, section_to_inject)

    with open("CHANGELOG.md", "w") as f:
        f.write(new_changelog)

if __name__ == "__main__":
    add_new_release_section_to_changelog()
