import re

path = "/root/DIT5411/train_all.log"
with open(path, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

# Collapse progress-bar carriage returns
lines = []
for line in text.splitlines():
    if "\r" in line:
        line = line.split("\r")[-1]
    lines.append(line)

print("=== Processes hint from log markers ===")
for line in lines:
    if line.startswith("=====") or line.startswith("Epoch ") or "Training complete" in line or "Saved" in line:
        print(line)

print("\n=== Last progress line ===")
progress = [ln for ln in lines if "/ " in ln and "loss:" in ln]
if progress:
    print(progress[-1][-300:])
else:
    print("No progress yet")

print("\n=== history_a.json if present ===")
try:
    import json
    with open("/root/DIT5411/results/history_a.json") as f:
        print(json.dumps(json.load(f), indent=2)[:800])
except Exception as e:
    print(e)
