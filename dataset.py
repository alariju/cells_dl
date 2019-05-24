import os

from PIL import Image
from torch.utils.data.dataset import Dataset


class CatsAndDogsDataset(Dataset):
    def __init__(self, directory, transform=None):
        """
        Inits the dataset
        :param directory: path to the directory with the samples
        :param transform: Pytorch transforms list
        """
        self.transform = transform
        (files, labels) = self.fill_files_and_labels(directory)
        self.labels = labels
        self.files = files

    def __getitem__(self, index):
        """
        :param index: dataset sample index
        :return: (sample, label)
        """
        file_path = self.files[index]
        # Open the image
        image_pil = Image.open(file_path)
        # apply transformations
        if self.transform is not None:
            tensor = self.transform(image_pil)
        # read the label
        y = self.labels[index]
        # transform label to numpy array
        # y = np.array([y]);
        return tensor, y

    def __len__(self):
        return len(self.files)

    @staticmethod
    def fill_files_and_labels(directory):
        """
        Extracts sample and label metadata
        :param directory: directory of samples
        :return: (file names, label array)
        """
        files = []
        labels = []
        # r=root, d=directories, f = files
        for root, directories, files in os.walk(directory):
            for file in files:
                if '.jpg' in file:
                    path = os.path.join(root, file)
                    files.append(path)
                    if 'cat' not in path:
                        labels += [0.0]
                    else:
                        labels += [1.0]
        return files, labels
