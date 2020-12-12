import numpy as np
from progress.bar import Bar
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DoublePendulumKinematicsDataset(Dataset):
    def __init__(self, l1: float, l2: float):
        super().__init__()

        self.l1 = l1
        self.l2 = l2

    def __getitem__(self, idx):
        phi = -np.pi + torch.rand(size=(2, )) * np.pi

        x2 = torch.sin(phi[0]) * self.l1 + torch.sin(phi[1]) * self.l2
        y2 = torch.cos(phi[0]) * self.l1 + torch.cos(phi[1]) * self.l2

        input = torch.tensor([x2, y2])

        target = phi

        # print("input", input)
        # print("target", target)

        return input, target


    def __len__(self):
        return 20000

class Net(nn.Module):
    def __init__(self, l1: float, l2: float):
        super().__init__()

        self.l1 = l1
        self.l2 = l2

        self.fc1 = nn.Linear(2, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(64, 2)
        self.relu4 = nn.ReLU()

        self.seq = nn.Sequential(self.fc1, self.relu1, self.fc2, self.relu2, self.fc3, self.relu3, self.fc4, self.relu4)

    def forward(self, input):
        x = input
        x[:, 0] /= self.l1
        x[:, 1] /= self.l2

        x = self.seq(input)

        if self.train is False:
            for i in range(input.size(0)):
                for j in range(input.size(1)):
                    if x[i, j] > np.pi:
                        while x[i, j] > np.pi:
                            x[i, j] -= 2*np.pi
                    elif x[i, j] < -np.pi:
                        while x[i, j] < -np.pi:
                            x[i, j] += 2*np.pi

        return x
    

def eval(dataloader, net):
    net.eval()
    with torch.no_grad():
        eval_loss = 0
        num_samples = 0
        progress_bar = Bar(f"Evaluate", max=len(dataloader))
        for batch_idx, (input, target) in enumerate(dataloader):
            pred = net(input)

            loss = F.l1_loss(pred, target)

            eval_loss += float(loss.item())
            num_samples += input.size(0)

            progress_bar.next()
        progress_bar.finish()

        eval_loss /= num_samples

        print(f"Evaluation mean l1 loss: {eval_loss*360} deg")


if __name__ == "__main__":
    np.random.seed(101)
    torch.manual_seed(101)

    l1, l2 = 0.5, 1

    dataset = DoublePendulumKinematicsDataset(l1=l1, l2=l2)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=6)

    net = Net(l1=l1, l2=l2)

    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(20):
        net.train()
        progress_bar = Bar(f"Train epoch {epoch}", max=len(dataloader))
        for batch_idx, (input, target) in enumerate(dataloader):
            optimizer.zero_grad()

            pred = net(input)

            loss = F.l1_loss(pred, target)

            if batch_idx % 100 == 0:
                pass
                # print(f"\nloss for batch {batch_idx}: {loss.item()*360} deg")

            loss.backward()
            optimizer.step()

            progress_bar.next()
        progress_bar.finish()

        eval(dataloader, net)

    