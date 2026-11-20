"""python -m creditlens.scenarios  -> write data/scenarios and print the two case files."""
import json

from creditlens.scenarios import case_file, generate_scenarios

if __name__ == "__main__":
    out = generate_scenarios()
    print(f"Wrote {out}")
    for app in ("A901", "A902"):
        cf = case_file(out, app)
        print(f"\n== {app} ({cf['company_id']}) ==")
        for f in cf["flags"]:
            print(f"- {f['code']}: {f['title']}")
        print(json.dumps({f["code"]: f["evidence"] for f in cf["flags"]}, indent=1, default=str)[:1500])
