import matplotlib.pyplot as plt
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
import torch
import torch.nn as nn
import numpy as np


def lifneuron(mem, spike, thresh=2, tau=20.0):
    decay = -mem/tau
    mem += decay + spike
    spk = (mem >= thresh).float()
    mem *= (1 - spk)
    return mem, spk
    

input_spike_train = torch.tensor([
    1, 1, 1, 0, 1, 0, 0, 1, 1, 1,
    0, 0, 1, 1, 0, 1, 0, 0, 0, 1 ], dtype=torch.float)

timesteps = len(input_spike_train)
mem = torch.tensor(0.0)
mem_trace = []
spk_trace = []

for t in range(timesteps):
    mem, spk = lifneuron(mem, input_spike_train[t])
    mem_trace.append(mem.item())
    spk_trace.append(spk.item())

print(f"mem potential at timestep {t}", mem_trace)
print(f"spike train at timestep {t}", spk_trace)
    
fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

# Input spikes
axs[0].stem(range(timesteps), input_spike_train, linefmt='C0-', markerfmt='C0o')
axs[0].set_ylabel("Input Spike")
axs[0].set_title("Input Spike Train")

# Membrane potential
axs[1].plot(range(timesteps), mem_trace, 'C1')
axs[1].axhline(1.0, color='gray', linestyle='--', label="Threshold")
axs[1].set_ylabel("Membrane Potential")
axs[1].set_title("Membrane Potential Over Time")
axs[1].legend()

# Output spikes
axs[2].stem(range(timesteps), spk_trace, linefmt='C2-', markerfmt='C2o')
axs[2].set_ylabel("Output Spike")
axs[2].set_xlabel("Time step")
axs[2].set_title("Output Spikes")

plt.tight_layout()
plt.show()


