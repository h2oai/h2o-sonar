#!/usr/bin/env python3
# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import re
import sys
from pathlib import Path

"""
Post-process changelog.rst to add unique labels to section headers.

This script fixes Sphinx duplicate label warnings by adding unique labels
to each version's subsections (Added, Fixed, Changed, etc.).

Usage:
    python make/fix_changelog_labels.py docs/source/changelog.rst
"""

def extract_version_from_header(line: str) -> str | None:
    """Extract version string from RST header line.

    Example: "`v3.0.0 <url>`__ — 2026/?/?" -> "v3-0-0"
    """
    match = re.search(r'`(v[\d\.]+)', line)
    if match:
        version = match.group(1)
        # convert dots to dashes for valid RST labels
        return version.replace('.', '-')
    return None


def fix_changelog_labels(rst_file: Path) -> None:
    """Add unique labels to changelog section headers."""
    with open(rst_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # section headers to fix (as they appear after pandoc conversion)
    sections = ['Added', 'Fixed', 'Changed', 'Security', 'Deprecated', 'Removed']

    # pattern to match old numeric labels (e.g., .. _added-1:)
    old_label_pattern = re.compile(r'^\.\. _(added|fixed|changed|security|deprecated|removed)-\d+:\s*$')

    output_lines = []
    current_version = None
    i = 0

    while i < len(lines):
        line = lines[i]

        # skip old numeric labels
        if old_label_pattern.match(line):
            # also skip following blank line if present
            if i + 1 < len(lines) and lines[i + 1].strip() == '':
                i += 2
                continue
            i += 1
            continue

        # detect version header (contains link with version)
        if '`v' in line and '<https://github.com/h2oai/h2o-sonar/tree/v' in line:
            current_version = extract_version_from_header(line)
            output_lines.append(line)
            i += 1
            continue

        # detect section headers that need modification
        stripped = line.strip()
        if current_version and stripped in sections:
            # check if next line is underline (tildes)
            if i + 1 < len(lines) and lines[i + 1].strip().startswith('~'):
                # modify section title to include version to avoid implicit label duplication
                # convert v3-0-0 back to v3.0.0 for display
                display_version = current_version.replace('-', '.')
                modified_header = f"{stripped} ({display_version})\n"
                # recalculate underline length
                underline = '~' * (len(stripped) + len(display_version) + 3) + '\n'
                output_lines.append(modified_header)
                output_lines.append(underline)
                output_lines.append('\n')  # add blank line after section header
                i += 2  # skip original header and underline
                # skip blank lines after original underline
                while i < len(lines) and lines[i].strip() == '':
                    i += 1
                continue

        output_lines.append(line)
        i += 1

    # write modified content back
    with open(rst_file, 'w', encoding='utf-8') as f:
        f.writelines(output_lines)

    print(f"Fixed labels in {rst_file}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python make/fix_changelog_labels.py docs/source/changelog.rst")
        sys.exit(1)

    rst_file = Path(sys.argv[1])
    if not rst_file.exists():
        print(f"Error: {rst_file} does not exist")
        sys.exit(1)

    fix_changelog_labels(rst_file)


if __name__ == '__main__':
    main()
