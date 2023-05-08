import os
import re
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from utils import check_exists, makedir_exist_ok, save, load

transform = transforms.Compose([
    # transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


# path = 'D:\\PythonProjects\\federateLearn\\data\\tinyimagnet'
# img_path = os.path.join(path, 'train', 'n03763968', 'images', 'n03763968_347.JPEG')
# print(img_path)
# img = Image.open(img_path)
# transform = transforms.ToTensor()
# img_tensor = transform(img)
# print(img_tensor)
# print(img.mode)


class TinyImagenet(Dataset):
    def __init__(self, path, split, subset, transform=None):  # split: train or val
        self.path = path
        self.subset = subset
        self.split = split
        self.transform = transform
        # self.classes = class2num(path)
        if not check_exists(self.processed_folder):
            self.process()
        self.img, self.target = load(os.path.join(self.processed_folder, '{}.pt'.format(self.split)))
        self.target = self.target[subset]
        self.classes = load(os.path.join(self.processed_folder, 'meta.pt'))
        self.classes_size = len(self.classes)

    def __getitem__(self, index):
        img, target = self.img[index], torch.tensor(self.target[index])
        return {'img': img, self.subset: target}

    def __len__(self):
        return len(self.target)

    @property
    def processed_folder(self):
        return os.path.join(self.path, 'processed')

    def process(self):
        train_set, test_set, meta = self.make_data()
        save(train_set, os.path.join(self.processed_folder, 'train.pt'))
        save(test_set, os.path.join(self.processed_folder, 'val.pt'))
        save(meta, os.path.join(self.processed_folder, 'meta.pt'))
        return

    def make_data(self):
        classes = class2num(self.path)
        train_img, train_label = read_dir(self.path, 'train', classes)
        test_img, test_label = read_dir(self.path, 'val', classes)
        train_target, test_target = {'label': train_label}, {'label': test_label}
        return (train_img, train_target), (test_img, test_target), classes


def class2num(path):
    path = os.path.join(path, 'wnids.txt')
    with open(path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    classes = {}
    for i, cls in enumerate(lines):
        classes[cls] = i
    return classes


def read_dir(path, filename, c2n):
    # classes = class2num(path)
    img, target = [], []
    path = os.path.join(path, filename)
    if filename == 'train':
        clsdir = os.listdir(path)
        for c in clsdir:
            clspath = os.path.join(path, c)
            imgdir = os.listdir(os.path.join(clspath, 'images'))
            for i in imgdir:
                imgpath = os.path.join(clspath, 'images', i)
                pimg = Image.open(imgpath).convert('RGB')
                img.append(transform(pimg))
                target.append(c2n[c])
        img = [torch.unsqueeze(x, dim=0) for x in img]
        img = torch.cat(img, dim=0)
    elif filename == 'val':
        target_path = os.path.join(path, 'val_annotations.txt')
        with open(target_path, 'r') as f:
            index = f.read()
        target = re.findall('(n\d+)', index)
        target = [c2n[s] for s in target]
        imgdir = os.listdir(os.path.join(path, 'images'))
        for im in imgdir:
            imgpath = os.path.join(path, 'images', im)
            pimg = Image.open(imgpath).convert('RGB')
            img.append(transform(pimg))
        img = [torch.unsqueeze(x, dim=0) for x in img]
        img = torch.cat(img, dim=0)
    else:
        raise ValueError('not find {}'.format(filename))
    return img, target


# print(os.listdir(path))
# dic = class2num(path)

# a, b = make_data(path, 'val', dic)
# print(a.shape, len(b))
# tiny = TinyImagenet(path, 'train', 'label')
# print(tiny[0]['img'].shape)
