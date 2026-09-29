"""
Independently check a model delivered by GPUSetu.

It loads the model on THIS laptop (no GPU needed) and re-tests it on MNIST's
10,000 test images, then compares the result with what the host claimed in
model_card.json. If the numbers match, the model is real and did the work.

Usage (from the gpu folder):
    python verify_model.py results\\<job_id>
"""

import json
import sys
from pathlib import Path

import torch
from torchvision import datasets

folder = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
sys.path.insert(0, str(folder))
from load_model import load          # the loader that came with the model

model = load(folder / "model.pt")
card = json.loads((folder / "model_card.json").read_text())
print(f"Loaded model.pt: {sum(p.numel() for p in model.parameters()):,} parameters")

test = datasets.MNIST("mnist_data", train=False, download=True)   # ~2 MB, first run only
x = test.data.float().div(255).unsqueeze(1)
y = test.targets

correct = 0
with torch.no_grad():
    for i in range(0, 10000, 1000):
        correct += (model(x[i:i + 1000]).argmax(1) == y[i:i + 1000]).sum().item()
accuracy = correct / 10000

with torch.no_grad():
    preds = model(x[:10]).argmax(1).tolist()
print("First 10 test images   real digits:", y[:10].tolist())
print("                  model predictions:", preds)

claimed = card["test_accuracy"]
print(f"\nAccuracy measured here : {accuracy:.2%}")
print(f"Accuracy host claimed  : {claimed:.2%}")
print("RESULT:", "MATCHES. The model is real." if abs(accuracy - claimed) < 0.002
      else "DOES NOT MATCH. Something is wrong with this model.")
