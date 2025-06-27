import snntorch as snn
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from snntorch import utils, spikegen, surrogate
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import wandb
import os

wandb.init(project="snn-mnist")
os.makedirs("outputs", exist_ok=True)

batch_size=128
data_path='/tmp/data/mnist'
num_classes = 10

dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

# Create DataLoaders
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)

# network
num_inputs = 28*28
num_hidden = 1000
num_outputs = 10
beta = 0.9
timesteps = 100
lr = 5e-4

class snNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)
    
    # len(timesteps) = len(spiketrain)
    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spktrain2 = []
        memtrain2 = []
        for step in range(timesteps):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spktrain2.append(spk2)
            memtrain2.append(mem2)
        return torch.stack(spktrain2), torch.stack(memtrain2)


net = snNet().to(device)
lossfn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

trainloss = []
testloss = []
epochs = 10
count = 0
prnt = 50


for ep in range(epochs):
    for i, (data, targ) in enumerate(train_loader):
        data = data.to(device)
        targ = targ.to(device)
        spikedata = spikegen.rate(data.view(data.size(0), -1), num_steps=timesteps)
        net.train() #can turn it off

        spkout, memout = net(spikedata)
        loss = sum([lossfn(memout[step], targ) for step in range(timesteps)])
        #backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        trainloss.append(loss.item())

        _, pred = spkout.sum(dim=0).max(1)
        train_acc = (pred == targ).float().mean().item()

        #eval
        net.eval()
        with torch.no_grad():
            testdata, testtarg = next(iter(test_loader))
            testspike = spikegen.rate(testdata.view(testdata.size(0), -1), num_steps=timesteps)
            testspk, testmem = net(testspike)
            testloss_val = sum([lossfn(testmem[step], testtarg) for step in range(timesteps)])
            testloss.append(testloss_val.item())
            _, testpred = testspk.sum(dim=0).max(1)
            testacc = (testpred == testtarg).float().mean().item()


        wandb.log({
            "train/loss": loss.item(),
            "train/accuracy": train_acc,
            "test/loss": testloss_val.item(),
            "test/accuracy": testacc,
            "step": count
        })

        if count % prnt == 0:
            print(f"Epoch {ep}, Iter {i}")
            print(f"Train Loss: {loss.item():.2f}, Acc: {train_acc*100:.2f}%")
            print(f"Test  Loss: {testloss_val.item():.2f}, Acc: {testacc*100:.2f}%\n")

        count += 1


net.eval()
correct = 0
total = 0
testloader2 = DataLoader(mnist_test, batch_size=batch_size, drop_last=False)
with torch.no_grad():
    for data, targets in testloader2:
        data = data.to(device)
        targets = targets.to(device)
        spike_input = spikegen.rate(data.view(data.size(0), -1), num_steps=timesteps)
        spk_out, _ = net(spike_input)
        _, predicted = spk_out.sum(dim=0).max(1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

final_acc = 100 * correct / total
wandb.log({"final/test_accuracy": final_acc})
wandb.finish()

print(f"Final Test Accuracy: {final_acc:.2f}%")

torch.save(net.state_dict(), "snn_mnist.pth")
print("Model saved to snn_mnist.pth")








