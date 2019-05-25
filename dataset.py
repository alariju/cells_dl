import os
import random

from PIL import Image
from torchvision import transforms as torch_transforms
from torchvision.transforms import functional
from torch.utils.data.dataset import Dataset


class CellsDataset(Dataset):
    def __init__(self, directory, transforms=None):
        self.common_transforms = transforms
        self.zero_transforms = torch_transforms.Compose([
            torch_transforms.RandomCrop(100)
        ])
        (zero_cells_files, with_cells_files) = self.fill_files_and_labels(
            directory)
        self.zero_cells_files = zero_cells_files
        self.with_cells_files = with_cells_files

    @staticmethod
    def fill_files_and_labels(_directory):
        zero_cells_files = []
        with_cells_files = []
        for root, directories, _ in os.walk(_directory):
            for directory in directories:
                for _, _, files in os.walk(os.path.join(root, directory)):
                    for file in files:
                        file_path = os.path.join(root, directory, file)
                        if 'zero_cells' in directory:
                            if 'gimp.jpg' in file_path:
                                zero_cells_files.append(file_path)
                        else:
                            with_cells_files.append(file_path)

        return zero_cells_files, with_cells_files

    def __getitem__(self, index):
        if random.random() < .5:
            label = 1.
            random_idx = random.randint(0, len(self.with_cells_files) - 1)
            path = self.with_cells_files[random_idx]
            pil_image = Image.open(path)
            pil_image = pil_image.convert('RGB')
        else:
            label = 0.
            random_idx = random.randint(0, len(self.zero_cells_files) - 1)
            path = self.zero_cells_files[random_idx]
            pil_image = Image.open(path)
            pil_image = self.zero_transforms(pil_image)
        # if index < len(self.with_cells_files):
        # else:
        tensor = self.common_transforms(pil_image)
        return tensor, label
        # if index < len(self.zero_cells_files):
        #     # index corresponds to a zero cells file
        #     label = 0.
        #     path = self.zero_cells_files[index]
        #     # Shouldn't apply rgb transform
        # else:
        #     label = 1.
        #     path = self.with_cells_files[index - len(self.zero_cells_files)]
        #
        # pil_image = Image.open(path)
        # pil_image = pil_image.convert(mode="RGB")
        # pil_image = functional.adjust_gamma(pil_image, gamma=0.8)
        # tensor = self.tv_transforms(pil_image)
        # return tensor, label

    def __len__(self):
        return len(self.with_cells_files) * 2
        # return len(self.with_cells_files) * 2
