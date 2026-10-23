"""Re-render the magazine from saved JSON, without a new (paid) AI generation.

Every /magazine run saves its content to weekly_blogs/last_magazine.json.

    python render_magazine.py [weekly_blogs/last_magazine.json] [out.pdf]
"""
import json
import logging
import sys

import magazine

logging.basicConfig(level=logging.INFO)
src = sys.argv[1] if len(sys.argv) > 1 else "weekly_blogs/last_magazine.json"
out = sys.argv[2] if len(sys.argv) > 2 else "magazine_rerender.pdf"
with open(src, encoding="utf-8") as fh:
    saved = json.load(fh)
print(magazine.render(saved["data"], out, source_articles=saved.get("source_articles")))
