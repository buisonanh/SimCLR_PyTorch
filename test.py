from models.resnet_simclr import ResNetSimCLR
import torch
import torch.nn as nn
import torchvision.datasets
from tqdm import tqdm
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset

state_dict = torch.load("checkpoints\checkpoint_0002.pth.tar")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ResNetSimCLR(base_model='resnet18', out_dim=128)
model.load_state_dict(state_dict['state_dict'])

for param in model.parameters():
    param.requires_grad = False

net = nn.Sequential(model, nn.Linear(128, 7))
net.to(device)

trainset = torchvision.datasets.ImageFolder("fer_plus/train", transform=ContrastiveLearningDataset.get_simclr_pipeline_transform(224))
testset = torchvision.datasets.ImageFolder("fer_plus/test", transform=ContrastiveLearningDataset.get_simclr_pipeline_transform(224))

train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=512, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=512, shuffle=False, num_workers=2)


optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)

for epoch in range(10):
    net.train()
    for i, (images, labels) in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(f"Epoch {epoch}, step {i}, loss {loss.item()}")

    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Epoch {epoch}, accuracy {100 * correct / total}")