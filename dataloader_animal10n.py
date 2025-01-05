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


def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

class animal10n_dataset(Dataset):
    def __init__(self,  dataset,  noise_type, noise_path, root_dir, transform, mode, transform_s=None, is_human=True, noise_file='',
                 pred=[], probability=[],probability2=[] ,log='', print_show=False, r =0.2 , noise_mode = 'cifarn'):
        assert dataset == 'animal10n'
        self.dataset = dataset
        self.transform = transform
        self.transform_s = transform_s
        self.mode = mode
        self.noise_type = noise_type
        self.noise_path = noise_path
        self.print_show = print_show
        self.noise_mode = noise_mode
        self.r = r
        self.nb_classes = 10

        train_folder = os.path.join(root_dir, 'training')
        test_folder = os.path.join(root_dir, 'testing')
        train_files = os.listdir(train_folder)
        test_files = os.listdir(test_folder)

        if self.mode == 'test':
            self.test_data = [np.asarray(Image.open(os.path.join(test_folder, i))) for i in test_files]
            self.test_label = [int(i.split('_')[0]) for i in test_files]
        else:
            self.train_data = [np.asarray(Image.open(os.path.join(train_folder, i))) for i in train_files]
            self.train_labels = [int(i.split('_')[0]) for i in train_files]
            # dummy place holder
            self.noise_or_not = np.transpose(self.train_labels) != np.transpose(self.train_labels)

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
                    clean = (np.array(self.noise_label) == np.array(self.train_labels))
                    log.write('Numer of labeled samples:%d   AUC (not computed):%.3f\n' % (pred.sum(), 0))
                    log.flush()
                elif self.mode == "unlabeled":
                    pred_idx = (1 - pred).nonzero()[0]
                self.train_data = self.train_data[pred_idx]
                self.noise_label = [self.noise_label[i] for i in pred_idx]
                self.print_wrapper("%s data has a size of %d" % (self.mode, len(self.noise_label)))
        self.print_show = False

    def print_wrapper(self, *args, **kwargs):
        if self.print_show:
            print(*args, **kwargs)

    def load_label(self):
        # NOTE only load manual training label
        noise_label = torch.load(self.noise_path)
        if isinstance(noise_label, dict):
            if "clean_label" in noise_label.keys():
                clean_label = torch.tensor(noise_label['clean_label'])
                assert torch.sum(torch.tensor(self.train_labels) - clean_label) == 0
                self.print_wrapper(f'Loaded {self.noise_type} from {self.noise_path}.')
                self.print_wrapper(f'The overall noise rate is {1 - np.mean(clean_label.numpy() == noise_label[self.noise_type])}')
            return noise_label[self.noise_type].reshape(-1)
        else:
            raise Exception('Input Error')

    def __getitem__(self, index):
        if self.mode == 'labeled':
            img, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.transform_s(img)
            return img1, img2, target, prob
        elif self.mode == 'unlabeled':
            img = self.train_data[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.transform_s(img)
            return img1, img2
        elif self.mode == 'all_lab':
            img, target, prob, prob2 = self.train_data[index], self.noise_label[index], self.probability[index],self.probability2[index]
            true_labels = self.train_labels[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.transform_s(img)
            return img1, img2, target, prob,prob2,true_labels, index
        elif self.mode == 'all':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            if self.transform_s is not None:
                img1 = self.transform(img)
                img2 = self.transform_s(img)
                return img1, img2, target, index
            else:
                img = self.transform(img)
                return img, target, index
        elif self.mode == 'all2':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.transform_s(img)
            return img1, img2, target, index
        elif self.mode == 'test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target

    def __len__(self):
        if self.mode != 'test':
            return len(self.train_data)
        else:
            return len(self.test_data)


class animal10n_dataloader():
    def __init__(self, dataset, noise_type, noise_path, is_human, batch_size, num_workers, root_dir, log,
                 noise_file='', noise_mode='cifarn', r=0.2):
        self.r = r
        self.noise_mode = noise_mode
        self.dataset = dataset
        self.noise_type = noise_type
        self.noise_path = noise_path
        self.is_human = is_human
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file

        self.transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        self.transform_train_s = copy.deepcopy(self.transform_train)
        self.transform_train_s.transforms.insert(0, RandomAugment(3, 5))
        self.transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.print_show = True

    def run(self, mode, pred=[], prob=[], prob2=[]):
        if mode == "warmup":
            all_dataset = animal10n_dataset(
                dataset=self.dataset,
                noise_type=self.noise_type,
                noise_path=self.noise_path,
                is_human=self.is_human,
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
            return trainloader, all_dataset.train_labels

        elif mode == "train":
            labeled_dataset = animal10n_dataset(
                dataset=self.dataset,
                noise_type=self.noise_type,
                noise_path=self.noise_path,
                is_human=self.is_human,
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

            return labeled_trainloader, labeled_dataset.train_labels

        elif mode == "test":
            test_dataset = animal10n_dataset(
                dataset=self.dataset,
                noise_type=self.noise_type,
                noise_path=self.noise_path,
                is_human=self.is_human,
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
            eval_dataset = animal10n_dataset(
                dataset=self.dataset,
                noise_type=self.noise_type,
                noise_path=self.noise_path,
                is_human=self.is_human,
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
            return eval_loader, eval_dataset.noise_or_not
        # never print again
