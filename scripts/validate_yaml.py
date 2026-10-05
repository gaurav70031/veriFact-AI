"""Validate YAML syntax of compose and config files."""
import yaml, sys, os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
files = [
    os.path.join(root, "docker-compose.yml"),
    os.path.join(root, "docker-compose.dev.yml"),
]

all_ok = True
for f in files:
    name = os.path.basename(f)
    try:
        with open(f) as fh:
            yaml.safe_load(fh)
        print(f"[OK]   {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {e}")
        all_ok = False

sys.exit(0 if all_ok else 1)
