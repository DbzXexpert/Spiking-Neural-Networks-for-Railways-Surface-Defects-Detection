import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

class SNN(nn.Module):
    def __init__(self, n_steps):
        super(SNN, self).__init__()
        self.n_steps = n_steps
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 1000)

        self.lif1 = snn.Leaky(beta=0.5, spike_grad=surrogate.fast_sigmoid())
        self.lif2 = snn.Leaky(beta=0.5, spike_grad=surrogate.fast_sigmoid())
        self.lif3 = snn.Leaky(beta=0.5, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk1_rec, spk2_rec, spk3_rec = [], [], []
        mem1_rec, mem2_rec, mem3_rec = [], [], []

        for step in range(self.n_steps):
            spk1, mem1 = self.lif1(self.conv1(x), mem1)
            spk1 = self.pool(spk1)
            spk1_rec.append(spk1)
            mem1_rec.append(mem1)

            spk2, mem2 = self.lif2(self.conv2(spk1), mem2)
            spk2 = self.pool(spk2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

            spk2_flat = spk2.view(spk2.size(0), -1)
            spk3, mem3 = self.lif3(self.fc1(spk2_flat), mem3)
            spk3_rec.append(spk3)
            mem3_rec.append(mem3)

        return torch.stack(spk1_rec, dim=0), torch.stack(mem1_rec, dim=0)

if __name__ == "__main__":
    print("SNN Backbone model class is ready.")
