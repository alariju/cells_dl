import os

from PIL import Image
from torchvision.transforms import functional
from torch.utils.data.dataset import Dataset


class CellsDataset(Dataset):
    def __init__(self, directory, tv_transforms=None):
        self.tv_transforms = tv_transforms
        (zero_cells_files, with_cells_files) = self.fill_files_and_labels(
            directory)
        self.zero_cells_files = zero_cells_files
        self.with_cells_files = with_cells_files

    @staticmethod
    def fill_files_and_labels(_directory):
        zero_cells_files = []
        with_cells_files = []
        # r=root, d=directories, f = files
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
        if index < len(self.zero_cells_files):
            # index corresponds to a zero cells file
            label = 0.
            path = self.zero_cells_files[index]
            # Shouldn't apply rgb transform
        else:
            label = 1.
            path = self.with_cells_files[index - len(self.zero_cells_files)]

        pil_image = Image.open(path)
        pil_image = pil_image.convert(mode="RGB")
        pil_image = functional.adjust_gamma(pil_image, gamma=0.8)
        tensor = self.tv_transforms(pil_image)
        return tensor, label

    def __len__(self):
        return len(self.zero_cells_files) + len(self.with_cells_files)
