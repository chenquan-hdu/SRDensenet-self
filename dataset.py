import torch.utils.data as data
from os import listdir
from os.path import join
from PIL import Image
from torchvision.transforms import Compose, ToTensor, CenterCrop


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    #img = Image.open(filepath).convert('RGB')
    y, _, _ = img.split()
    return y

def transform():
    return Compose([
        #CenterCrop(128),
        ToTensor()
    ])

def get_training_set():
    root_dir = 'D:/project/SRDenseNet-self/data/train/'
    LR_dir = join(root_dir, "LR")
    HR_dir = join(root_dir, "HR")

    return DatasetFromFolder(LR_dir, HR_dir, input_transform=transform(), target_transform=transform())


def get_test_set():
    root_dir = 'D:/project/SRDenseNet-self/data/test/'
    LR_dir = join(root_dir, "LR")
    HR_dir = join(root_dir, "HR")

    return DatasetFromFolder(LR_dir, HR_dir, input_transform=transform(), target_transform=transform())


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir_1, image_dir_2, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames_1 = [join(image_dir_1, x) for x in listdir(image_dir_1) if is_image_file(x)]
        self.image_filenames_2 = [join(image_dir_2, x) for x in listdir(image_dir_2) if is_image_file(x)]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames_1[index])
        target = load_img(self.image_filenames_2[index])

        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames_1)
