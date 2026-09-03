"""Smoke tests for the forecasting service. Run: python test_app.py"""
import json
import app as svc

c = svc.app.test_client()
fails = 0

def check(label, cond, extra=""):
    global fails
    print(f"  {'PASS' if cond else 'FAIL'}  {label} {extra}")
    if not cond:
        fails += 1

print("\n/health")
h = c.get("/health").get_json()
check("status ok", h["status"] == "ok")
check("series count > 0", h["n_series"] > 0, f"({h['n_series']})")
check("primary method is naive", h["primary_method"] == "naive")

print("\n/series")
s = c.get("/series").get_json()
check("returns a list", s["count"] > 0, f"({s['count']} series)")
demo = s["series"][0]

print(f"\n/forecast  ({demo['market']} / {demo['commodity']})")
for h_ in (1, 3, 6):
    r = c.post("/forecast", json={**demo, "horizon": h_})
    j = r.get_json()
    ok = r.status_code == 200 and j["forecast_kes_per_kg"] > 0
    check(f"horizon={h_}", ok,
          f"-> {j.get('forecast_kes_per_kg')} "
          f"[{j.get('interval_80', {}).get('low')}, {j.get('interval_80', {}).get('high')}]")
    if ok and h_ > 1:
        prev = c.post("/forecast", json={**demo, "horizon": h_ - 1}).get_json()
        wider = ((j["interval_80"]["high"] - j["interval_80"]["low"])
                 > (prev["interval_80"]["high"] - prev["interval_80"]["low"]))
        check(f"  interval widens at h={h_}", wider)

print("\nerror handling")
check("unknown series -> 404",
      c.post("/forecast", json={"market": "Nowhere", "commodity": "X"}).status_code == 404)
check("missing field -> 400", c.post("/forecast", json={"market": "X"}).status_code == 400)
check("horizon 0 -> 400", c.post("/forecast", json={**demo, "horizon": 0}).status_code == 400)
check("horizon 99 -> 400", c.post("/forecast", json={**demo, "horizon": 99}).status_code == 400)
check("empty body -> 400", c.post("/forecast", json={}).status_code == 400)

print(f"\n{'ALL PASSED' if fails == 0 else str(fails) + ' FAILED'}")
raise SystemExit(1 if fails else 0)
