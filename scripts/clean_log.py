path = "/root/DIT5411/train_all.log"
with open(path, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()
# Keep only last segment after each carriage return on a line
lines = []
for line in text.splitlines():
    if "\r" in line:
        line = line.split("\r")[-1]
    lines.append(line)

keys = ("Epoch", "accuracy", "Error", "Traceback", "Saved", "Training complete", "=====", "loss:", "val_")
for line in lines:
    if any(k in line for k in keys):
        print(line)
