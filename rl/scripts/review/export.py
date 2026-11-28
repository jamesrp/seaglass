"""Collect the numbers behind the review's two charts into REVIEW_DIR/chartdata.json."""
import collections, glob, json, os, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metaprobe import HUMAN
# The folder the logs are in (`watch.py` writes them to REVIEW_DIR/games).
S = os.environ.get("REVIEW_DIR", "review")

def rows(mode, ctrl=None):
    by = collections.defaultdict(list)
    n = 0
    for f in sorted(glob.glob(f"{S}/games/{mode}_*.json")):
        g = json.load(open(f)); n += 1
        for p, pl in g["players"].items():
            if ctrl and g["controllers"][int(p)] != ctrl:
                continue
            for t in pl["turns"]:
                by[t["turn"]].append(t)
    return by, n

def q(v, p):
    v = sorted(v); return v[min(len(v) - 1, int(p * len(v)))]

out = {"tier": {}, "stats": {}, "n": {}}
for key, mode, ctrl in [("plain", "selfplain", None), ("search", "allsearch", None), ("gamer", "bots", "9kgamerbot"), ("sovereign", "slow", "sovereign"), ("forced", "curve-standard", "curve"), ("slowagent", "slow", "agent")]:
    by, n = rows(mode, ctrl)
    out["n"][key] = n
    out["tier"][key] = {t: round(st.mean(x["tier"] for x in by[t]), 2) for t in range(1, 17) if len(by[t]) >= 8}
    tot = {t: [sum(u["atk"] + u["hp"] for u in x["board"]) for x in by[t]] for t in range(1, 19) if len(by[t]) >= 8}
    out["stats"][key] = {t: {"med": st.median(v), "p90": q(v, 0.9), "max": max(v), "n": len(v)} for t, v in tot.items()}
    out.setdefault("rolls", {})[key] = {t: round(st.mean(sum(a.startswith("roll") for a in x["actions"]) for x in by[t]), 1) for t in range(1, 17) if len(by[t]) >= 8}
    out.setdefault("hp", {})[key] = {t: round(st.mean(x["health"] for x in by[t]), 1) for t in range(1, 17) if len(by[t]) >= 8}
# pure run at iteration 750, from the Mac's ladder file
d = json.load(open(os.environ.get("PURE_EVAL", "runs/pure/evals/iter_00750.json")))
who = [k for k in d["by_turn"] if "bot" not in k.lower() and k not in ("tempo", "heuristic", "semirandom", "greedylevel")]
print("pure by_turn keys:", list(d["by_turn"].keys()))
k = who[0] if who else list(d["by_turn"].keys())[0]
out["tier"]["pure"] = {int(t): round(v["tier"], 2) for t, v in d["by_turn"][k].items() if int(t) <= 16 and v.get("n", 99) >= 8}
out["human"] = [{"comp": name, "turn": t, "total": sum(a + h for _, a, h, _ in spec), "top": max(a + h for _, a, h, _ in spec)} for name, t, spec in HUMAN]
json.dump(out, open(f"{S}/chartdata.json", "w"))
for k in out["tier"]:
    print(k, out["n"].get(k), out["tier"][k])
for k in out["stats"]:
    print(k, {t: int(v["med"]) for t, v in out["stats"][k].items() if t >= 8}, "p90", {t: int(v["p90"]) for t, v in out["stats"][k].items() if t in (12, 13, 14, 15)}, "max", {t: int(v["max"]) for t, v in out["stats"][k].items() if t in (12, 13, 14, 15)})
print(out["human"])
print("rolls", out["rolls"]["plain"], out["rolls"]["gamer"])
