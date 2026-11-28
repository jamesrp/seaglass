"""Pull each card's rules text from the engine's doc comments into REVIEW_DIR/cardtext.json.

    python cardtext.py [CARD NAME ...]
"""
import os, re, sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3] / "engine" / "src"
S = os.environ.get("REVIEW_DIR", "review")
TITLE_RE = re.compile(r"^`([^`]+)`\s*(?:\([^)]*\))?\s*[—-]?\s*(.*)$")
out = {}
for sub in ["cards/minions","cards/spells","trinkets","dark_gifts","darkmoon","heroes"]:
    for f in (ROOT/sub).rglob('*.rs'):
        lines=[]
        for line in f.read_text().splitlines():
            if not line.startswith('//!'): break
            lines.append(line[3:].strip())
        if not lines: continue
        m = TITLE_RE.match(lines[0])
        if not m: continue
        para=[]
        for l in lines[1:]:
            if not l:
                if para: break
                continue
            para.append(l)
        out[m.group(1)] = re.sub(r"[*`]","", (m.group(2)+' '+' '.join(para))).strip()
json.dump(out, open(f'{S}/cardtext.json','w'), indent=0)
print(len(out))
for n in sys.argv[1:]:
    print(f"* {n}: {out.get(n,'?')}")
