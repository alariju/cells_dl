import os

import torch
from PIL import Image
from torch import nn, optim
from torch.autograd import Variable
from torchvision import transforms

from cnn import ResNET50
from dataset import CellsDataset


def create_model(learning_rate, device):
    model = ResNET50().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    return model, optimizer, criterion


def train_model(device, dataset, epochs, learning_rate, batch_size, weights_file):
    # create the model with the optimizer and the loss function
    (model, optimizer, criterion) = create_model(learning_rate, device)
    dataset_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        for batch_id, (images, labels) in enumerate(dataset_loader):
            images = Variable(images)
            labels = Variable(labels)
            labels = labels.long()
            # save labels and samples to the device
            inputs, labels = images.to(device), labels.to(device)
            # put weights to zer
            optimizer.zero_grad()

            # estimated output by the model
            output = model.forward(inputs)

            # calculate model loss
            loss = criterion(output, labels)
            # back propagate error
            loss.backward()
            # update weigths
            optimizer.step()

            # train mode
            model.train()

            if batch_id % 100 == 0:
                print(f"epoch: {epoch} [{batch_id * len(images)}/"
                      f"{len(dataset_loader.dataset)} "
                      f"({100. * batch_id / len(dataset_loader):.0f}%)"
                      f"\tLoss: {loss.item():.6f}]")
                # print('Epoch de entrenamiento: {} [{}/{} ({
                # :.0f}%)]\tPerdida: {:.6f}'.format( epochs, batch_id * len(
                # train), len(dataset_loader.dataset), 100. * batch_id /
                # len(dataset_loader), loss.item())) guarda el modelo
                torch.save(model.state_dict(), weights_file)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device: ")
    print(device)

    train_transforms = transforms.Compose(
        [transforms.RandomRotation(30),
         transforms.RandomResizedCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225])])
    dataset = CellsDataset("datasets/cells/train", train_transforms)

    epochs = 100
    learning_rate = 0.001
    batch_size = 65
    weights_file = "weights/weights_file"
    train_model(
        device,
        dataset,
        epochs,
        learning_rate,
        batch_size,
        weights_file
    )


def convert_0_cells_to_rgb():
    directory = "datasets/cells/train/0_cells"
    for root, directories, files in os.walk(directory):
        for file in files:
            image = Image.open(os.path.join(root, file))
            image = image.convert("RGB")
            image.save(os.path.join(root, file), "TIFF")


if __name__ == '__main__':
    main()
