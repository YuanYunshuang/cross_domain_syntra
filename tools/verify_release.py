# -*- coding: utf-8 -*-
"""Check a downloaded SYNTRA release against its SHA256SUMS files.

Every `SHA256SUMS` found under the given directory is checked. Each line is
`<sha256>  <path>`, with the path relative to the folder that holds that SHA256SUMS. Files
that are listed but missing, and files whose hash differs, are reported. With --strict, files
that sit next to a SHA256SUMS but are not listed in any of them are reported too.

    python tools/verify_release.py ~/Downloads/syntra_release
    python tools/verify_release.py ~/Downloads/syntra_release --strict

Exit status: 0 if every listed file is present and matches, 1 otherwise.
"""

import argparse
import hashlib
import os
import sys

SUMS = "SHA256SUMS"


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def read_sums(sums_path):
    entries = []
    with open(sums_path) as f:
        for n, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            digest, sep, rel = line.partition("  ")
            if not sep or len(digest) != 64:
                raise SystemExit(f"{sums_path}:{n}: not '<sha256>  <path>': {line!r}")
            entries.append((digest.lower(), rel.lstrip("*")))
    return entries


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", help="release folder (or a single SHA256SUMS file)")
    p.add_argument("--strict", action="store_true", help="also report files no SHA256SUMS lists")
    args = p.parse_args()

    if os.path.isfile(args.root):
        sums_files = [os.path.abspath(args.root)]
    else:
        sums_files = sorted(os.path.join(d, SUMS) for d, _, files in os.walk(args.root) if SUMS in files)
    if not sums_files:
        raise SystemExit(f"no {SUMS} under {args.root}")

    listed, ok, bad, missing = set(), 0, [], []
    for sums in sums_files:
        base = os.path.dirname(sums)
        entries = read_sums(sums)
        for digest, rel in entries:
            path = os.path.normpath(os.path.join(base, rel))
            listed.add(path)
            if not os.path.isfile(path):
                missing.append(path)
            elif sha256(path) != digest:
                bad.append(path)
            else:
                ok += 1
        print(f"{sums}: {len(entries)} entries")

    unlisted = []
    if args.strict:
        for sums in sums_files:
            for d, _, files in os.walk(os.path.dirname(sums)):
                for f in files:
                    path = os.path.normpath(os.path.join(d, f))
                    if f != SUMS and path not in listed:
                        unlisted.append(path)

    for label, paths in (("MISSING", missing), ("MISMATCH", bad), ("UNLISTED", sorted(set(unlisted)))):
        for path in paths:
            print(f"  {label:9s}{path}")
    print(f"{ok} ok, {len(bad)} mismatched, {len(missing)} missing"
          + (f", {len(set(unlisted))} unlisted" if args.strict else ""))
    sys.exit(0 if not bad and not missing and not (args.strict and unlisted) else 1)


if __name__ == "__main__":
    main()
