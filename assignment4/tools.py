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


# KNearestNeighbor Classifier from Assignment 1
class KNearestNeighbor(object):
    """ a kNN classifier using L2 distance """
    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train classifier for k nearest neibor
        Parameters
        ----------
        X : numpy array
            shape(num_train, D) training data images
        Y : numpy array shape (N,)
            labels for images X
        """
        self.X_train = X
        self.y_train = y
        
    def predict(self, X, k=1):
        """
        Predict labels for test data using this classifier.
        Parameters
        ----------
        X : numpy array 
            of shape (num_test, D) containing test data consisting
            of num_test samples each of dimension D.
        k : float
            The number of nearest neighbors that vote for the predicted labels.

        Returns
        -------
        y : numpy array
            of shape (num_test,) containing predicted labels for the
            test data, where y[i] is the predicted label for the test point
            X[i].  
        """
        
        dists = self.compute_distances(X)
        return self.predict_labels(dists, k=k)

    def compute_distances(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.
        
        Parameters
        ----------
        X : numpy array
            test images of shape (num_test, D)
        
        Returns
        -------
        dists : numpy array
            distances from each training image to each test image
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train)) 

        dists = np.reshape(np.sum(X**2, axis=1), [num_test,1]) + np.sum(self.X_train**2, axis=1) - 2 * np.matmul(X, self.X_train.T)
        
        dists = np.sqrt(dists)

        return dists

    def k_neighbors_idx(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.
        Parameters
        ----------
        dists : numpy array 
            of shape (num_test, num_train) where dists[i, j]
            gives the distance betwen the ith test point and the jth training
            point.
        Returns
        -------
        y : numpy array
            of shape (num_test,) containing predicted labels for the
            test data, where y[i] is the predicted label for the test point
            X[i].  
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []

            closest_X = self.X_train[np.argsort(dists[i])][0:k]
            y_pred[i] = np.bincount(closest_y.astype(int)).argmax()

        return y_pred

        