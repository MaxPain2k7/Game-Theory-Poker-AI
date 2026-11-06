"""Bake a trained Leduc strategy into the browser game.

    python build_web.py strategies/leduc_cfrplus.json

Writes docs/index.html, a single self-contained page you can open locally or
host on GitHub Pages.
"""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("strategy", nargs="?", default="strategies/leduc_cfrplus.json")
    args = ap.parse_args()

    payload = json.loads(Path(args.strategy).read_text())
    if payload["game"] != "leduc":
        raise SystemExit("the browser game plays classic 6 card Leduc, pass a leduc strategy")
    table = {k: {a: round(p, 4) for a, p in row.items()} for k, row in payload["strategy"].items()}
    body = Path("docs/template.html").read_text().replace("__STRATEGY__", json.dumps(table, separators=(",", ":")))
    page = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            "</head>\n<body>\n" + body + "\n</body>\n</html>\n")
    Path("docs/index.html").write_text(page)
    print("wrote docs/index.html")


if __name__ == "__main__":
    main()
