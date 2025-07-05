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
        spk2_rec, spk1_rec = [], []
        mem2_rec, mem1_rec = [], []
        

        for step in range(timesteps):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)
            spk1_rec.append(spk1)
            mem1_rec.append(mem1)

        return (
        torch.stack(spk1_rec),
        torch.stack(mem1_rec),
        torch.stack(spk2_rec),
        torch.stack(mem2_rec),
    )

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
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)


os.makedirs("outputs", exist_ok=True)

with torch.no_grad():
    data, targets = next(iter(test_loader))
    data, targets = data.to(device), targets.to(device)
    spike_input = spikegen.rate(data.view(data.size(0), -1), num_steps=timesteps)
    spk1, mem1, spk2, mem2 = model(spike_input)
    _, predicted = spk2.sum(dim=0).max(1)


    raw_img = data[0].cpu().numpy().squeeze()
    spike_input = spike_input.cpu()
    spk1, mem1 = spk1.cpu(), mem1.cpu()
    spk2, mem2 = spk2.cpu(), mem2.cpu()
    weights = {
        "fc1": model.fc1.weight.data.cpu().numpy().tolist(),
        "fc2": model.fc2.weight.data.cpu().numpy().tolist()
    }

    export = {
        "meta": {
            "encoding": "rate",
            "timesteps": timesteps,
            "arch": {
                "input": 784,
                "fc1": 1000,
                "fc2": 10,
                "activation": "LIF",
            }
        },
        "raw_image": raw_img.tolist(),
        "spike_input": spike_input[:, 0, :].tolist(),
        "spk1": spk1[:, 0, :].tolist(),
        "mem1": mem1[:, 0, :].tolist(),
        "spk2": spk2[:, 0, :].tolist(),
        "mem2": mem2[:, 0, :].tolist(),
        "label": int(targets[0]),
        "prediction": int(predicted[0]),
        "weights": weights
    }

    import json, os
    os.makedirs("viz_out", exist_ok=True)
    with open("viz_out/snndata_full.json", "w") as f:
        json.dump(export, f, indent=2)
    print("Exported full SNN data to viz_out/snndata_full.json")


    '''
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
    '''

