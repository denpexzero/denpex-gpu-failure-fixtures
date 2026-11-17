#!/usr/bin/env python3
"""Validate schema, unique IDs, and upstream references in fixtures/corpus-sample-25.json."""

import json
import sys
from pathlib import Path
from urllib.parse import urlparse

REQUIRED_FIELDS = (
    "failure_id",
    "source_url",
    "source_type",
    "failure_type",
    "title",
    "framework",
    "environment",
    "error_message",
)


def main() -> int:
    fixture_path = Path(__file__).parent / "fixtures" / "corpus-sample-25.json"
    if not fixture_path.exists():
        print(f"ERROR: Missing fixture file at {fixture_path}", file=sys.stderr)
        return 1

    records = json.loads(fixture_path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) < 20:
        print("ERROR: Expected a JSON list of at least 20 fixtures", file=sys.stderr)
        return 1

    seen_ids: set[str] = set()
    failure_types: set[str] = set()

    for idx, rec in enumerate(records):
        for field in REQUIRED_FIELDS:
            val = rec.get(field)
            if not isinstance(val, str) or not val.strip():
                print(f"ERROR: Record {idx} missing non-empty field '{field}'", file=sys.stderr)
                return 1

        fid = rec["failure_id"]
        if fid in seen_ids:
            print(f"ERROR: Duplicate failure_id '{fid}'", file=sys.stderr)
            return 1
        seen_ids.add(fid)

        parsed = urlparse(rec["source_url"])
        if parsed.scheme != "https" or not parsed.netloc:
            print(f"ERROR: Record '{fid}' has invalid HTTPS source_url '{rec['source_url']}'", file=sys.stderr)
            return 1

        failure_types.add(rec["failure_type"])

    print(
        f"Verified {len(records)} public GPU failure fixtures across {len(failure_types)} failure types."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
