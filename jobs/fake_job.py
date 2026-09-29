"""
GPUSetu FRAUD SIMULATION — what a dishonest host might do.

It pretends to run the buyer's training job: prints convincing progress lines and a
fake "DONE" result, but never actually trains on the GPU. The host then claims
payment for the full time. The marketplace's anomaly agent should catch this,
because the GPU telemetry shows the graphics card was idle.

Used only for the live fraud demo.
"""

import random
import time

FAKE_SECONDS = 30

print("GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU", flush=True)
print("Data: real MNIST (60,000 training images, 10,000 unseen test images)", flush=True)

start = time.time()
step = 0
loss = 0.09
while time.time() - start < FAKE_SECONDS:
    time.sleep(5)
    step += random.randint(180, 210)
    loss *= random.uniform(0.55, 0.85)
    print(f"t={time.time() - start:5.1f}s  step={step}  loss={loss:.4f}", flush=True)

print(f"DONE: {step} steps in {time.time() - start:.1f}s, test accuracy={random.uniform(98.9, 99.3):.2f}%", flush=True)
