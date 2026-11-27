"""Where do the Tier 5+ minions on a seat's board come from, and what does a seat buy once it is on Tier 5/6?"""
import collections, glob, json, os, sys
import seaglass
# The folder the logs are in (`watch.py` writes them to REVIEW_DIR/games).
S = os.environ.get("REVIEW_DIR", "review")
cat = seaglass.catalog()
TIER = {}
for c in cat:
    try:
        TIER[c.name] = c.tavern_tier
    except Exception:
        TIER[c["name"]] = c["tier"]

ENGINES = ["Brann Bronzebeard", "Titus Rivendare", "Drakkari Enchanter", "Balinda Stonehearth", "Living Azerite", "Felboar", "Auto Reveille", "Polarizing Beatboxer", "Spark Snapper", "Leyline Surfacer", "Tasty Lobster", "Deathstrider", "Headhunter Gryphon", "Forsaken Weaver", "Snazzy Phantom", "Dead Bellringer", "Glambot", "Young Murk-Eye", "Persistent Poet", "Hopebringer", "Nadina the Red", "Gatekeeper Amalgam", "N'raqi Frostcaller", "Hot-Air Surveyor", "De-volition-ist", "Bile Spitter", "Dark Paradox", "Green Volumizer", "Felfire Conjurer", "Bronze Timewalker", "Runic Arcanist", "Sky Admiral Rogers", "Enterprising Escapee", "Kalecgos, Arcane Aspect", "Leeroy the Reckless", "Highkeeper Ra", "Crimson Vindicator", "Heroic Broodmother"]


def run(mode, ctrl=None, turns=(12,)):
    src = collections.Counter()
    buys_on_tier = collections.defaultdict(collections.Counter)  # tavern tier -> minion tier of buys
    eng = collections.Counter()
    nseat = 0
    nfinal = 0
    for f in sorted(glob.glob(f"{S}/games/{mode}_*.json")):
        g = json.load(open(f))
        for p, pl in g["players"].items():
            if ctrl and g["controllers"][int(p)] != ctrl:
                continue
            bought, disc_dg, disc_other, tr = set(), set(), set(), set()
            prev_tier = 1
            for t in pl["turns"]:
                tier_now = prev_tier
                last = None
                for a in t["actions"]:
                    verb, _, rest = a.partition(" ")
                    if verb.startswith("LEVEL->"):
                        tier_now = int(verb[7:8])
                    elif verb == "buy":
                        bought.add(rest)
                        mt = TIER.get(rest.replace("Golden ", ""))
                        if mt:
                            buys_on_tier[tier_now][mt] += 1
                    elif verb == "discover":
                        name = rest.split(" (of")[0]
                        (disc_dg if last == "dark_discovery" else disc_other).add(name)
                    last = verb
                prev_tier = t["tier"]
                if t["turn"] in turns:
                    nseat += 1
                    for u in t["board"]:
                        if u["tier"] >= 5:
                            n = u["name"].replace("Golden ", "")
                            if n in bought:
                                src["bought in the Tavern"] += 1
                            elif n in disc_dg:
                                src["Dark Gift"] += 1
                            elif n in disc_other:
                                src["other Discover (triple reward, spells, heroes)"] += 1
                            else:
                                src["other (trinket, summon, transform)"] += 1
            if pl["turns"]:
                nfinal += 1
                names = {u["name"].replace("Golden ", "") for u in pl["turns"][-1]["board"]}
                for e in ENGINES:
                    if e in names:
                        eng[e] += 1
    tot = sum(src.values())
    print(f"== {mode}/{ctrl}: Tier 5+ minions on boards at turn {turns}: {tot} over {nseat} seat-turns ({tot / max(1, nseat):.2f} per board)")
    for k, v in src.most_common():
        print(f"   {k}: {v / max(1, tot):.0%}")
    print("   buys by Tavern Tier at the time (share of minion buys that are Tier 5+, n):")
    for tt in sorted(buys_on_tier):
        c = buys_on_tier[tt]
        n = sum(c.values())
        print(f"     Tavern Tier {tt}: {sum(v for k, v in c.items() if k >= 5) / n:.0%} T5+, {sum(v for k, v in c.items() if k >= tt) / n:.0%} at the top tier offered, {sum(v for k, v in c.items() if k <= 2) / n:.0%} T1-2, n={n}")
    print(f"   final boards with the card ({nfinal} seats): " + "; ".join(f"{k} {v / nfinal:.1%}" for k, v in eng.most_common()))
    print("   never seen: " + ", ".join(e for e in ENGINES if e not in eng))


if __name__ == "__main__":
    mode = sys.argv[1]
    ctrl = sys.argv[2] if len(sys.argv) > 2 else None
    run(mode, ctrl, turns=(12,))
    run(mode, ctrl, turns=(15,))
