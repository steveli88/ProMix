import os
import torch
import copy
import random
import json
from utils.randaug import *
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from math import exp, sqrt, log10
from Asymmetric_Noise import noisify_cifar100_asymmetric


def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


class clothing1m_dataset(Dataset):
    def __init__(self, dataset, noise_type, noise_path, root_dir, transform, mode, transform_s=None,
                 noise_file='', pred=[], probability=[], probability2=[], log='', print_show=False, r=0.2, noise_mode='cifarn'):
        assert dataset == 'clothing1m'
        self.dataset = dataset
        self.transform = transform
        self.transform_s = transform_s
        self.mode = mode
        self.noise_type = noise_type
        self.noise_path = noise_path
        self.print_show = print_show
        self.noise_mode = noise_mode
        self.r = r

        self.root_dir = root_dir
        self.train_labels = {}
        self.test_labels = {}

        with open('%s/noisy_label_kv.txt' % self.root_dir, 'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                img_path = entry[0][7:]
                self.train_labels[img_path] = int(entry[1])
        with open('%s/clean_label_kv.txt' % self.root_dir, 'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                img_path = entry[0][7:]
                self.test_labels[img_path] = int(entry[1])

        self.nb_classes = 14

        if self.mode == 'test':
            self.test_data_path = []
            with open('%s/clean_test_key_list.txt' % self.root_dir, 'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = l[7:]
                    self.test_data_path.append(img_path)
        else:
            self.train_data_path = []
            with open('%s/noisy_train_key_list.txt' % self.root_dir, 'r') as f:
                lines = f.read().splitlines()
                for i,l in enumerate(lines):
                    img_path = l[7:]
                    self.train_data_path.append(img_path)

            if self.mode == 'all_lab':
                self.probability = probability
                self.probability2 = probability2
                self.noise_label = self.train_labels
            elif self.mode == 'all':
                self.noise_label = self.train_labels
            else:
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]
                elif self.mode == "unlabeled":
                    pred_idx = (1 - pred).nonzero()[0]
                self.train_data_path = self.train_data_path[pred_idx]
                self.noise_label = self.train_labels

    def __getitem__(self, index):
        if self.mode == 'labeled':
            img_path = self.train_data_path[index]
            target = self.train_labels[img_path]
            prob = self.probability[index]
            img = Image.open(os.path.join(self.root_dir, 'images', img_path))
            img = img.convert('RGB')
            img1 = self.transform(img)
            img2 = self.transform_s(img)
            return img1, img2, target, prob
        elif self.mode == 'unlabeled':
            img_path = self.train_data_path[index]
            img = Image.open(os.path.join(self.root_dir, 'images', img_path))
            img = img.convert('RGB')
            img1 = self.transform(img)
            img2 = self.transform_s(img)
            return img1, img2
        elif self.mode == 'all_lab':
            img_path = self.train_data_path[index]
            target = self.train_labels[img_path]
            prob, prob2 = self.probability[index], self.probability2[index]
            true_labels = self.train_labels[img_path]
            img = Image.open(os.path.join(self.root_dir, 'images', img_path))
            img = img.convert('RGB')
            img1 = self.transform(img)
            img2 = self.transform_s(img)
            return img1, img2, target, prob, prob2, true_labels, index
        elif self.mode == 'all':
            img_path = self.train_data_path[index]
            target = self.train_labels[img_path]
            img = Image.open(os.path.join(self.root_dir, 'images', img_path))
            img = img.convert('RGB')
            if self.transform_s is not None:
                img1 = self.transform(img)
                img2 = self.transform_s(img)
                return img1, img2, target, index
            else:
                img = self.transform(img)
                return img, target, index
        elif self.mode == 'all2':
            img_path = self.train_data_path[index]
            target = self.train_labels[img_path]
            img = Image.open(os.path.join(self.root_dir, 'images', img_path))
            img = img.convert('RGB')
            img1 = self.transform(img)
            img2 = self.transform_s(img)
            return img1, img2, target, index
        elif self.mode == 'test':
            img_path = self.test_data_path[index]
            target = self.test_labels[index]
            img = Image.open(os.path.join(self.root_dir, 'images', img_path))
            img = img.convert('RGB')
            img = self.transform(img)
            return img, target

    def __len__(self):
        if self.mode != 'test':
            return len(self.train_data_path)
        else:
            return len(self.test_data_path)


class clothing1m_dataloader():
    def __init__(self, dataset, noise_type, noise_path, batch_size, num_workers, root_dir, log,
                 noise_file='', noise_mode='cifarn', r=0.2):
        self.r = r
        self.noise_mode = noise_mode
        self.dataset = dataset
        self.noise_type = noise_type
        self.noise_path = noise_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file

        self.transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
        ])
        self.transform_train_s = copy.deepcopy(self.transform_train)
        self.transform_train_s.transforms.insert(0, RandomAugment(3, 5))
        self.transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
        ])

        self.print_show = True

    def run(self, mode, pred=[], prob=[], prob2=[]):
        if mode == "warmup":
            all_dataset = clothing1m_dataset(
                dataset=self.dataset,
                noise_type=self.noise_type,
                noise_path=self.noise_path,
                root_dir=self.root_dir,
                transform=self.transform_train,
                transform_s=self.transform_train_s,
                mode="all",
                noise_file=self.noise_file,
                print_show=self.print_show,
                r=self.r,
                noise_mode=self.noise_mode,
            )
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )
            self.print_show = False
            # never show noisy rate again
            return trainloader, list(all_dataset.train_labels.values())

        elif mode == "train":
            labeled_dataset = clothing1m_dataset(
                dataset=self.dataset,
                noise_type=self.noise_type,
                noise_path=self.noise_path,
                root_dir=self.root_dir,
                transform=self.transform_train,
                mode="all_lab",
                noise_file=self.noise_file,
                pred=pred,
                probability=prob,
                probability2=prob2,
                log=self.log,
                transform_s=self.transform_train_s,
                r=self.r,
                noise_mode=self.noise_mode,
            )
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True,
            )

            return labeled_trainloader, list(labeled_dataset.train_labels.values())

        elif mode == "test":
            test_dataset = clothing1m_dataset(
                dataset=self.dataset,
                noise_type=self.noise_type,
                noise_path=self.noise_path,
                root_dir=self.root_dir,
                transform=self.transform_test,
                mode="test",
                r=self.r,
                noise_mode=self.noise_mode,
            )
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            return test_loader

        elif mode == "eval_train":
            eval_dataset = clothing1m_dataset(
                dataset=self.dataset,
                noise_type=self.noise_type,
                noise_path=self.noise_path,
                root_dir=self.root_dir,
                transform=self.transform_test,
                mode="all",
                noise_file=self.noise_file,
                r=self.r,
                noise_mode=self.noise_mode,
            )
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            return eval_loader

