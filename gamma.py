# script to compute firing rate of each layer
# for energy consumption computation


import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import snntorch as snn
from snntorch import spikegen

num_inputs = 28 * 28
num_hidden = 1000
num_outputs = 10
beta = 0.9
timesteps = 100 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class snNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk1_rec = []
        spk2_rec = []

        for step in range(timesteps):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk1_rec.append(spk1)
            spk2_rec.append(spk2)

        return torch.stack(spk1_rec), torch.stack(spk2_rec)

model = snNet().to(device)
model.load_state_dict(torch.load("snn_mnist.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))
])

data_path = "/tmp/data/mnist"
test_dataset = datasets.MNIST(root=data_path, train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

total_spikes_fc1 = 0
total_spikes_fc2 = 0
total_possible_fc1 = 0
total_possible_fc2 = 0

with torch.no_grad():
    for data, targets in test_loader:
        data = data.to(device)
        batch_size = data.size(0)
        spike_input = spikegen.rate(data.view(batch_size, -1), num_steps=timesteps)

        spk1, spk2 = model(spike_input)

        # spk1: [timesteps, batch_size, num_hidden]
        # spk2: [timesteps, batch_size, num_outputs]

        total_spikes_fc1 += spk1.sum().item()
        total_spikes_fc2 += spk2.sum().item()

        total_possible_fc1 += batch_size * num_hidden * timesteps
        total_possible_fc2 += batch_size * num_outputs * timesteps

# Calculate γ
gamma_fc1 = total_spikes_fc1 / total_possible_fc1
gamma_fc2 = total_spikes_fc2 / total_possible_fc2

print("======================================")
print(f"Computed Firing Rates over {timesteps} timesteps:")
print(f"γ (firing rate) for fc1: {gamma_fc1:.6f}")
print(f"γ (firing rate) for fc2: {gamma_fc2:.6f}")
print("======================================")
