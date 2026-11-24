import time
import numpy as np

from harmonix.input.adapter import InputAdapter
from harmonix.live_audio import LiveAudioCoordinator, LiveAudioConfig

a = InputAdapter(device_index=1)
c = LiveAudioCoordinator(a, config=LiveAudioConfig())

block = np.zeros(2048, dtype=np.float32)

times_normal = []

for i in range(100):
    t0 = time.perf_counter()
    c.process_block(block)
    times_normal.append((time.perf_counter() - t0) * 1000)

print("NORMAL RECORDING")
print("FIRST 10 MEAN MS:", np.mean(times_normal[:10]))
print("LAST 10 MEAN MS:", np.mean(times_normal[-10:]))
print("MAX MS:", np.max(times_normal))

c2 = LiveAudioCoordinator(a, config=LiveAudioConfig())

c2._record_input = lambda block: None

times_no_record = []

for i in range(100):
    t0 = time.perf_counter()
    c2.process_block(block)
    times_no_record.append((time.perf_counter() - t0) * 1000)

print("\nRECORDING DISABLED")
print("FIRST 10 MEAN MS:", np.mean(times_no_record[:10]))
print("LAST 10 MEAN MS:", np.mean(times_no_record[-10:]))
print("MAX MS:", np.max(times_no_record))
