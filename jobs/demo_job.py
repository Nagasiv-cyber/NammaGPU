"""
GPUSetu demo job — trains a small image-recognition model on MNIST
(handwritten digits) for about 30 seconds so the GPU graph visibly climbs.

At the end it saves what the buyer paid for into OUTPUT_DIR (the sandbox's only
writable folder):
  model.pt          the trained weights
  load_model.py     a ready-to-run script to load and use the model
  model_card.json   what was trained, how long, and how accurate it is
"""

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

TARGET_SECONDS = 30      # how long to keep the GPU busy
BATCH_SIZE = 512
DATA_DIR = Path(__file__).parent / "data"


def load_data(device):
    """Real MNIST if available (pre-download it!), otherwise random fake images."""
    try:
        from torchvision import datasets
        ds = datasets.MNIST(DATA_DIR, train=True, download=False)   # never download during a demo
        test = datasets.MNIST(DATA_DIR, train=False, download=False)
        x = ds.data.float().div(255).unsqueeze(1)   # 60000 training images: 1 x 28 x 28 each
        y = ds.targets
        xt = test.data.float().div(255).unsqueeze(1)   # 10000 images the model NEVER trains on
        yt = test.targets
        print("Data: real MNIST (60,000 training images, 10,000 unseen test images)", flush=True)
    except Exception as e:
        print(f"Data: MNIST unavailable ({e}) -> using synthetic data", flush=True)
        x = torch.rand(60000, 1, 28, 28)
        y = torch.randint(0, 10, (60000,))
        xt, yt = x[:10000], y[:10000]
    return x.to(device), y.to(device), xt.to(device), yt.to(device)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)   # 28x28 -> 14x14
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)   # 14x14 -> 7x7
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def main():
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU found. Refusing to run on CPU.", flush=True)
        sys.exit(1)

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    x, y, xt, yt = load_data(device)
    model = SmallCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    start = time.time()
    last_print = start
    steps = 0
    while time.time() - start < TARGET_SECONDS:
        idx = torch.randint(0, x.shape[0], (BATCH_SIZE,), device=device)
        loss = F.cross_entropy(model(x[idx]), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        steps += 1
        if time.time() - last_print >= 5:
            print(f"t={time.time() - start:5.1f}s  step={steps}  loss={loss.item():.4f}", flush=True)
            last_print = time.time()

    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, 10000, 1000):          # check 1000 images at a time
            preds = model(xt[i:i + 1000]).argmax(1)       # score on UNSEEN test images
            correct += (preds == yt[i:i + 1000]).sum().item()
    acc = correct / 10000
    train_seconds = time.time() - start
    save_outputs(model, steps, train_seconds, acc)
    print(f"DONE: {steps} steps in {train_seconds:.1f}s, test accuracy={acc:.2%}", flush=True)


LOADER = '''"""Load the model trained on GPUSetu and read a handwritten digit.

Usage:  python load_model.py            (checks the model loads)
In code:  from load_model import load;  model = load("model.pt")
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def load(path="model.pt"):
    model = SmallCNN()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.eval()


def predict(model, image_28x28):
    """image_28x28: a 28x28 tensor of values 0..1 (white digit on black, like MNIST)."""
    with torch.no_grad():
        return int(model(image_28x28.reshape(1, 1, 28, 28).float()).argmax(1))


if __name__ == "__main__":
    m = load()
    print("Model loaded:", sum(p.numel() for p in m.parameters()), "parameters")
    print("Prediction on a blank image:", predict(m, torch.zeros(28, 28)))
'''


def save_outputs(model, steps, train_seconds, accuracy):
    """Write the buyer's deliverables. Skipped if no OUTPUT_DIR was given."""
    out = os.environ.get("OUTPUT_DIR")
    if not out:
        print("No OUTPUT_DIR set: model not saved.", flush=True)
        return
    try:
        out = Path(out)
        torch.save(model.cpu().state_dict(), out / "model.pt")
        (out / "load_model.py").write_text(LOADER)
        (out / "model_card.json").write_text(json.dumps({
            "task": "handwritten digit recognition (MNIST, 10 classes)",
            "architecture": "SmallCNN: 2 conv layers (64, 128 filters) + 2 dense layers",
            "parameters": sum(p.numel() for p in model.parameters()),
            "training_steps": steps,
            "batch_size": BATCH_SIZE,
            "training_seconds": round(train_seconds, 1),
            "test_accuracy": round(accuracy, 4),
            "test_set": "10,000 MNIST images never seen during training",
            "trained_on": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown",
            "framework": f"PyTorch {torch.__version__}",
            "how_to_load": "python load_model.py   or   from load_model import load; model = load('model.pt')",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, indent=2))
        size_mb = (out / "model.pt").stat().st_size / 1e6
        print(f"SAVED: model.pt ({size_mb:.1f} MB), load_model.py, model_card.json", flush=True)
    except Exception as e:
        print(f"WARNING: could not save outputs ({e})", flush=True)


if __name__ == "__main__":
    main()
