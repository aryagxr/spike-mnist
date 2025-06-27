import snntorch as snn
import torch
from torchvision import datasets, transforms
from snntorch import utils, spikegen
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
from IPython.display import HTML

import time
from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.table import Table
from rich.text import Text
import time


console = Console()
neuron_index = 210


batch_size=128
data_path='/tmp/data/mnist'
num_classes = 10

dtype = torch.float

transform = transforms.Compose([
            transforms.Resize((28,28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)

subset = 10
mnist_train = utils.data_subset(mnist_train, subset)
print(f"The size of mnist_train is {len(mnist_train)}")

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)

# spike train using rate coding
time_steps = 100
raw_vector = torch.ones(time_steps) * 0.5
print(raw_vector)
rate_code = torch.bernoulli(raw_vector)
print(rate_code)
print(f"The output is spiking {rate_code.sum()*100/len(rate_code):.2f}% of the time.")

data = iter(train_loader)
data_it, targets_it = next(data)

spike_data = spikegen.rate(data_it, num_steps=time_steps)
print(spike_data.shape)

spike_data_sample = spike_data[:, 0, 0]
print(spike_data_sample.shape)

num_steps = spike_data.shape[0]
image_index = 0   # Which image in the batch you want to visualize
spike_sample = spike_data[:, image_index, 0]  # [T, 28, 28]
spike_flat = spike_sample.reshape(num_steps, -1)  # [T, 784]

fig, ax = plt.subplots()
anim = splt.animator(spike_data_sample, fig, ax)

# Save as video
anim.save("spike_mnist.mp4")  # or "spike_mnist.gif"
print("Animation saved to spike_mnist.mp4")


def render_spike_grid(frame):
    table = Table.grid()
    for row in frame:
        row_str = "".join("█" if px.item() else " " for px in row)
        table.add_row(row_str)
    return Panel(table, title="Input Spike Grid (28x28)", border_style="bold cyan")

with Live(render_spike_grid(spike_sample[0]), refresh_per_second=10, screen=True) as live:
    for t in range(num_steps):
        live.update(render_spike_grid(spike_sample[t]))
        time.sleep(0.05)



