#!/usr/bin/env python3
# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json
import sys
from pathlib import Path

"""
Fix Pygments lexer in Jupyter notebooks from 'ipython3' to 'python3'.

This script modifies notebook metadata to prevent Sphinx warnings about
unknown 'ipython3' lexer.

Usage:
    python make/fix_notebook_lexer.py examples/predictive/*.ipynb
"""

def fix_notebook_lexer(notebook_path: Path) -> bool:
    """Fix pygments_lexer in notebook metadata.

    Returns:
        True if changes were made, False otherwise
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # check if fix is needed
    language_info = nb.get('metadata', {}).get('language_info', {})
    current_lexer = language_info.get('pygments_lexer')

    if current_lexer == 'ipython3':
        # change ipython3 to python3
        language_info['pygments_lexer'] = 'python3'

        # write back
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write('\n')  # add trailing newline

        print(f"Fixed: {notebook_path}")
        return True
    elif current_lexer == 'python3':
        print(f"Already correct: {notebook_path}")
        return False
    else:
        print(f"Skipped (lexer={current_lexer}): {notebook_path}")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python make/fix_notebook_lexer.py path/to/notebooks/*.ipynb")
        sys.exit(1)

    notebook_paths = [Path(arg) for arg in sys.argv[1:]]

    fixed_count = 0
    for nb_path in notebook_paths:
        if nb_path.exists() and nb_path.suffix == '.ipynb':
            if fix_notebook_lexer(nb_path):
                fixed_count += 1
        else:
            print(f"Not found or not a notebook: {nb_path}")

    print(f"\nTotal: {len(notebook_paths)} notebooks, {fixed_count} fixed")


if __name__ == '__main__':
    main()
