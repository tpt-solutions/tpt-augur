#!/usr/bin/env python3
"""Verify that every checked-off item in TODO.md maps to real code.

This is the guard described in TODO.md Phase 0: the checklist must not silently
drift from the codebase. Every line `- [x]` in TODO.md must carry an inline
marker of the form:

    - [x] Short description <!-- verify: path/to/file.rs -->
    - [x] Another item       <!-- verify: file.rs::symbol_name, other.rs -->

The script checks that each referenced file exists relative to the repository
root, and (when `::symbol` is given) that the symbol appears in that file.
Unmarked checked items are treated as failures so drift cannot hide.
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TODO = os.path.join(ROOT, "TODO.md")

CHECKED_RE = re.compile(r"^\s*-\s*\[[xX]\]\s*(.*)$")
VERIFY_RE = re.compile(r"<!--\s*verify:\s*(.*?)\s*-->")
SYMBOL_SPLIT_RE = re.compile(r"\s*,\s*")


def fail(msg):
    print(f"FAIL: {msg}")
    return False


def check_target(target):
    """Return (ok, message) for a single `file` or `file::symbol` target."""
    if "::" in target:
        path, symbol = target.split("::", 1)
        path = path.strip()
        symbol = symbol.strip()
    else:
        path, symbol = target.strip(), ""

    full = os.path.join(ROOT, path)
    if not os.path.isfile(full):
        return False, f"missing file: {path}"

    if symbol:
        with open(full, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        # Allow a symbol to appear as a fn/struct/enum/trait/mod name.
        if not re.search(r"\b" + re.escape(symbol) + r"\b", text):
            return False, f"symbol `{symbol}` not found in {path}"
    return True, f"ok: {target}"


def main():
    if not os.path.isfile(TODO):
        print(f"FAIL: {TODO} not found")
        return 1

    with open(TODO, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    checked = 0
    ok = 0
    problems = 0

    # Markdown list items may wrap across several indented continuation lines;
    # accumulate the full text of each bullet before testing it.
    item_lines: list[str] = []
    item_start = 0

    def finalize():
        nonlocal checked, ok, problems, item_lines
        if not item_lines:
            return
        first = item_lines[0]
        m = CHECKED_RE.match(first)
        if not m:
            item_lines = []
            return
        lineno = item_start
        checked += 1
        joined = "".join(item_lines)
        vm = VERIFY_RE.search(joined)
        if not vm:
            problems += 1
            print(f"FAIL: line {lineno}: checked item has no `<!-- verify: ... -->` marker")
            item_lines = []
            return

        targets = [t for t in SYMBOL_SPLIT_RE.split(vm.group(1).strip()) if t]
        if not targets:
            problems += 1
            print(f"FAIL: line {lineno}: empty verify marker")
            item_lines = []
            return

        item_ok = True
        for t in targets:
            good, msg = check_target(t)
            prefix = "  ok  " if good else "  FAIL"
            print(f"{prefix}: line {lineno}: {msg}")
            if not good:
                item_ok = False
        if item_ok:
            ok += 1
        else:
            problems += 1
        item_lines = []

    for lineno, raw in enumerate(lines, start=1):
        if raw.lstrip().startswith("- "):
            finalize()
            item_lines = [raw]
            item_start = lineno
        elif raw.strip() == "":
            finalize()
        elif item_lines:
            item_lines.append(raw)
    finalize()

    print(
        f"\n{checked} checked item(s): {ok} verified, {problems} problem(s)."
    )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
