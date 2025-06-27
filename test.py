import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import snntorch as snn
from snntorch import spikegen
import matplotlib.pyplot as plt
import os


num_inputs = 28 * 28
num_hidden = 1000
num_outputs = 10
beta = 0.9
timesteps = 10
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
        spk2_rec = []

        for step in range(timesteps):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)

        return torch.stack(spk2_rec)


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
test_dataset = datasets.MNIST(root=data_path, train=False, transform=transform, download=False)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)


os.makedirs("outputs", exist_ok=True)

with torch.no_grad():
    data, targets = next(iter(test_loader))
    data, targets = data.to(device), targets.to(device)
    spike_input = spikegen.rate(data.view(data.size(0), -1), num_steps=timesteps)
    spk_out = model(spike_input)
    _, predicted = spk_out.sum(dim=0).max(1)

    for idx in range(data.size(0)):
        plt.imshow(data[idx].cpu().squeeze(), cmap='gray')
        plt.title(f"Predicted: {predicted[idx].item()} | Label: {targets[idx].item()}")
        plt.axis("off")
        plt.savefig(f"outputs/prediction_{idx}.png")
        plt.close()
        
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for idx, ax in enumerate(axes.flat):
        ax.imshow(data[idx].cpu().squeeze(), cmap='gray')
        ax.set_title(f"Pred: {predicted[idx].item()} | True: {targets[idx].item()}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("outputs/prediction_grid.png")
    plt.show()

