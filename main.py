import torch
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
from torchvision import transforms

from cnn import ResNET50, DenseNET121
from dataset import CellsDataset


def create_model(learning_rate, device):
    model = DenseNET121().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    return model, optimizer, criterion


def train_model(device, dataset, epochs, learning_rate, batch_size,
                weights_file):
    (model, optimizer, criterion) = create_model(learning_rate, device)
    dataset_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True)
    errors = []
    for epoch in range(epochs):
        for batch_id, (images, labels) in enumerate(dataset_loader):
            images = Variable(images)
            labels = Variable(labels)
            labels = labels.long()
            inputs, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            model.train()
            if batch_id % 100 == 0:
                loss_value = loss.item()
                print(f"epoch: {epoch} [{batch_id * len(images)}/"
                      f"{len(dataset_loader.dataset)} "
                      f"({100. * batch_id / len(dataset_loader):.0f}%)"
                      f"\tLoss: {loss_value:.6f}]")
                torch.save(model.state_dict(), weights_file)
                errors.append(loss_value)
    print(f"Median: {np.median(errors):.4f}")
    print(f"Std Deviation: {np.std(errors):.4f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ")
    print(device)
    train_transforms = transforms.Compose(
        [
            # transforms.RandomRotation(30, expand=True),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    dataset = CellsDataset("datasets/cells/train", train_transforms)

    epochs = 5
    learning_rate = 0.001
    batch_size = 50
    weights_file = "weights/weights_file"
    train_model(
        device,
        dataset,
        epochs,
        learning_rate,
        batch_size,
        weights_file
    )


if __name__ == '__main__':
    main()
