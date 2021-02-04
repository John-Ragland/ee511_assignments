import numpy as np
import matplotlib.pyplot as plt
import torch

def get_point():
    label = None
    x1 = 0
    x2 = 0
    while label is None:
        x1 = 2 * np.random.rand() - 1.0
        x2 = 2 * np.random.rand() - 1.0
        if np.abs(x1) + np.abs(x2) > 0.4 and np.abs(x1) + np.abs(x2) < 0.7:
            label =  0
        elif np.sqrt(np.square(x1) + np.square(x2)) < 0.3:
            label = 1
        elif np.sin(10.0 * x2) < 0:
            label = 2
        elif np.sin(5.0 * x1) > 0:
            label = 3

    return x1, x2, label


def generate_data(num_pts):
    x1_data = np.zeros(num_pts, dtype=float)
    x2_data = np.zeros(num_pts, dtype=float)
    labels = np.zeros(num_pts, dtype=int)
    for i in range(num_pts):
        x1_data[i], x2_data[i], labels[i] = get_point()

    return x1_data, x2_data, labels

def plot(x1, x2, labels):
    colors = ['r', 'g', 'b', 'y']
    for i in range(len(labels)):
        plt.plot(x1[i], x2[i], '+', color=colors[labels[i]])
    plt.show()

def train(model, device, train_loader, optimizer, epochs, log_interval, criterion, verbose=False):
    for epoch in range(epochs + 1):
        model.train()
        for batch_idx, (data, label) in enumerate(train_loader):
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label) 
            loss.backward()
            optimizer.step()
            if verbose and batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            test_loss += criterion(output, label) 
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return 100. * correct / len(test_loader.dataset)

def average_loss(model, criterion, imgs):
    return np.mean(loss_array(model, criterion, imgs))

def loss_array(model, criterion, imgs):
    loss = np.zeros(len(imgs))
    for i, im in enumerate(imgs):
        im_tensor = torch.Tensor(im)
        ouput = model(im_tensor)
        loss[i] = criterion(ouput, im_tensor)
    return loss