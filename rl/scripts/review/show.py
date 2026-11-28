"""Print one seat's game turn by turn.   python show.py FILE.json [PLAYER|winner|all-summary]"""

import json
import sys

r = json.load(open(sys.argv[1]))
who = sys.argv[2] if len(sys.argv) > 2 else "winner"
short = "--short" in sys.argv


def fmt_unit(u):
    n = u["name"].replace("Golden ", "")
    s = f"{n}{'*' if u.get('golden') else ''}[T{u['tier']}] {u['atk']}/{u['hp']}"
    kw = "".join({"divine_shield": "D", "reborn": "R", "taunt": "T", "venomous": "V", "windfury": "W", "stealth": "S"}[k] for k in u.get("kw", []))
    if kw:
        s += " " + kw
    if u.get("gifts"):
        s += " {" + ",".join(u["gifts"]) + "}"
    return s


def show(p):
    pl = r["players"][str(p)]
    print(f"### seed {r['seed']} seat {p} ({r['controllers'][p]}), hero {pl['hero']}, place {r['placements'][p]}, lobby types {r['tribes']}, game {r['turns']} turns")
    for t in pl["turns"]:
        acts = t["actions"]
        if short:
            import collections

            c = collections.Counter(a.split(" ")[0] for a in acts)
            lv = [a for a in acts if a.startswith("LEVEL")]
            a_s = f"rolls {c.get('roll', 0)}, buys {c.get('buy', 0)}, sells {c.get('sell', 0)}" + (", " + "; ".join(lv) if lv else "") + (", dark gift" if c.get("dark_discovery") else "")
        else:
            a_s = "; ".join(acts)
        tr = f" trinkets={t['trinkets']}" if t["trinkets"] else ""
        print(f"T{t['turn']:>2} tier {t['tier']} gold left {t['gold_left']} hp {t['health']}{tr}")
        print("     board: " + " | ".join(fmt_unit(u) for u in t["board"]))
        print("     did:   " + a_s)


if who == "winner":
    show(r["placements"].index(1))
elif who == "summary":
    for p in range(8):
        pl = r["players"][str(p)]
        last = pl["turns"][-1]
        tiers = {t["turn"]: t["tier"] for t in pl["turns"]}
        print(f"seat {p} {r['controllers'][p]:<16} place {r['placements'][p]} hero {pl['hero']:<28} out T{last['turn']:>2} tier@6/9/12: {tiers.get(6, '-')}/{tiers.get(9, '-')}/{tiers.get(12, '-')} final tier {last['tier']}")
        print("     final: " + " | ".join(fmt_unit(u) for u in last["board"]))
else:
    show(int(who))
