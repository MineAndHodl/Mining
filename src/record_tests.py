"""Run the offline test suite and record the result for the Model Room (data/test_status.json).
    python3 src/record_tests.py      -> prints the RESULT line; exit code = the suite's
"""
import os, re, json, subprocess, sys, datetime

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
r = subprocess.run([sys.executable, "tests/test_pipeline.py", "--offline"], cwd=ROOT, capture_output=True, text=True, env={**os.environ, "ODDS_API_KEY": ""})
line = next((l.strip() for l in r.stdout.splitlines() if "RESULT:" in l), "RESULT: unknown")
m = re.search(r"RESULT: (\d+)/(\d+)", line)
json.dump({"result": line, "passed": int(m.group(1)) if m else None, "total": int(m.group(2)) if m else None,
           "at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="minutes")}, open(os.path.join(ROOT, "data", "test_status.json"), "w"), indent=1)
print(line)
sys.exit(r.returncode)
