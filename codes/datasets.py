import torch

import pandas as pd

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils import data
from torch.utils.data import Subset, TensorDataset

from torch.distributions.multivariate_normal import MultivariateNormal


from datasets import load_dataset
from transformers import BertTokenizer
from transformers import AutoTokenizer


from enum import auto
from codes import DictEnum


import itertools 



from collections import defaultdict
import random as random


class Dataset(DictEnum):
    Normal = auto()
    MNIST = auto()
    CIFAR10 = auto()
    CIFAR100 = auto()
    GoEmotions = auto()


class Normal(data.Dataset):
    def __init__(self, cfg, train):
        self.dim = 10
        # self.nsamples = cfg.nsamples
        self.nsamples = 1000
        self.npeers = cfg.npeers
        self.mu = cfg.hratio
        if self.mu is None:
            self.mu = 0.5  # no metter

        # if self.npeers < 150:
        #     raise RuntimeError('npeers < 150')

        if train is False:
            self.test()
        elif train is True:
            self.train()

    def test(self):
        mean = 0.0 * torch.ones(self.dim)
        std = torch.eye(self.dim)
        dist = MultivariateNormal(mean, std)
        self.dset = dist.sample((self.nsamples,))

    def train(self):
        target_rank_below = 5
        # self.trueweights = torch.zeros(self.npeers)
        # self.trueweights[:target_rank_below] = 1 / target_rank_below

        self.trueweights = [1 / target_rank_below if i < target_rank_below else 0 for i in range(self.npeers)]

        data = list()
        std = torch.eye(self.dim)
        for rank in range(self.npeers):
            if rank < target_rank_below:
                mean = 0.0 * torch.ones(self.dim)

                dist = MultivariateNormal(mean, std)
                data.append(dist.sample((self.nsamples,)))
                # print(f"{data[-1].mean()=}")

            elif target_rank_below <= rank and rank < 100:
                mean = torch.ones(self.dim)
                mean /= torch.norm(mean)
                mean *= self.mu

                dist = MultivariateNormal(mean, std)
                data.append(dist.sample((self.nsamples,)))
            else:
                s = slice(0, self.dim, 2)
                mean = torch.ones(self.dim)
                mean[s] = 0
                mean /= torch.norm(mean)
                # mean /= 2

                dist = MultivariateNormal(mean, std)
                data.append(dist.sample((self.nsamples,)))

        self.dset = torch.cat(data)

    def split_train(self):
        indices_split = [[j + i*self.nsamples for j in range(self.nsamples)] for i in range(self.npeers)]
        return [Subset(self, inds) for inds in indices_split], self.trueweights, (self.dim,)

    def split_test(self, mdnsamples):
        indices = [i for i in range(self.nsamples)]

        val_inds = indices[:mdnsamples]
        test_inds = indices[mdnsamples:]

        return Subset(self, val_inds), None #Subset(self, test_inds)

    # def __len__(self):
    #     return len(self.dset)

    def __getitem__(self, idx):
        return 0, self.dset[idx]


def MNIST(cfg, train):
    root = '/tmp'
    transform = transforms.ToTensor()
    dataset = datasets.MNIST(root, train=train, transform=transform, download=True)
    return dataset


def CIFAR10(cfg, train):
    root = '/tmp'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    dataset = datasets.CIFAR10(root, train=train, transform=transform, download=True)
    return dataset


def CIFAR100(cfg, train):
    root = '/tmp'
    transform = transforms.ToTensor()  # add extra transforms
    dataset = datasets.CIFAR100(root, train=train, download=True, transform=transform)
    return dataset


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class GoEmotions(metaclass=Singleton):
    def __init__(self, cfg, train):
        self.cfg = cfg
        self.classes = True

        self.dset = load_dataset("go_emotions", "simplified")
        classes = self.dset["train"].features['labels'].feature.names

        self.idx2emotion = {i: t for i, t in enumerate(classes)}
        self.emotion2idx = {t: i for i, t in enumerate(classes)}

        self.dset.set_format(type="pandas")
#         train_df = emotion['train'][:]
#         valid_df = emotion['validation'][:]
#         test_df = emotion['test'][:]

#         train_df = train_df.groupby('label').apply(lambda x: x.sample(350)).reset_index(drop=True)
#         valid_df = valid_df.groupby('label').apply(lambda x: x.sample(70)).reset_index(drop=True)
#         test_df = test_df.groupby('label').apply(lambda x: x.sample(50)).reset_index(drop=True)

#         self.classes = train_df.label.unique()

        # PRETRAINED_LM = "bert-base-uncased"
        # self.tokenizer = BertTokenizer.from_pretrained(PRETRAINED_LM, do_lower_case=True)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

        self.mapping = {
            "anger": ["anger", "annoyance", "disapproval"],
            "disgust": ["disgust"],
            "fear": ["fear", "nervousness"],
            "joy": ["joy", "amusement", "approval", "excitement", "gratitude",  "love", "optimism", "relief", "pride", "admiration", "desire", "caring"],
            "sadness": ["sadness", "disappointment", "embarrassment", "grief",  "remorse"],
            "surprise": ["surprise", "realization", "confusion", "curiosity"],
            "neutral": ["neutral"]
        }
        self.classes = [e for e in itertools.chain.from_iterable(self.mapping.values())]
        # self.classes = list(self.mapping.keys())
        # self.bmap = {k: self.classes.index(k) for k in self.classes}
        # print(f"{self.bmap=}")
#         train_input_ids, train_att_masks = self.encode(train_df['text'].values.tolist())

#         valid_input_ids, valid_att_masks = self.encode(valid_df['text'].values.tolist())
#         test_input_ids, test_att_masks = self.encode(test_df['text'].values.tolist())

#         train_y = torch.LongTensor(train_df['label'].values.tolist())
#         valid_y = torch.LongTensor(valid_df['label'].values.tolist())
#         test_y = torch.LongTensor(test_df['label'].values.tolist())
#         train_y.size(),valid_y.size(),test_y.size()

#         print(f"Must implement trimming of test and val to {cfg.nclasses=}")

#         self.train = TensorDataset(train_input_ids, train_att_masks, train_y)
#         self.val = TensorDataset(valid_input_ids, valid_att_masks, valid_y)
#         self.test = TensorDataset(test_input_ids, test_att_masks, test_y)

    def tokenize_function(examples):
        return self.tokenizer(examples["text"], padding="max_length", truncation=True)



    # @staticmethod
    def encode(self, docs):
        '''
        This function takes list of texts and returns input_ids and attention_mask of texts
        '''
        # encoded_dict = self.tokenizer.batch_encode_plus(docs, add_special_tokens=True, max_length=128, padding='max_length', return_attention_mask=True, truncation=True, return_tensors='pt')
        encoded_dict = self.tokenizer(docs, add_special_tokens=True, max_length=128, padding='max_length', return_attention_mask=True, truncation=True, return_tensors='pt')

        input_ids = encoded_dict['input_ids']
        # print(f"{type(input_ids)=}")
        attention_masks = encoded_dict['attention_mask']
        # print(f"{type(attention_masks)=}")
        return input_ids, attention_masks

#     def split_train(self):
#         nclasses = self.cfg.nclasses
#         if nclasses != 28:
#             raise RuntimeError(f"nclasses != 3")
#         npeers = self.cfg.npeers
#         if npeers > 23:
#             raise RuntimeError(f"npeers > 23")

#         hratio = self.cfg.hratio

#         df = self.dset['train'][:]
#         mask = df['labels'].apply(lambda x: len(x)==1)
#         df = df.loc[mask]
#         df.loc[:, 'labels'] = df['labels'].apply(lambda x: x[0])

#         target_rank_below, near_target_rank_below = 2, 11
#         trueweights = [1 / target_rank_below if i < target_rank_below else 0 for i in range(npeers)]

#         print(f"{trueweights=}")
#         df_split = [pd.DataFrame() for _ in range(npeers)]

#         nsamples = 750
#         if nsamples > 1250:
#             raise RuntimeError(f"nsamples>1250")

#         if hratio is None:
#             hratio = 0.99  # no matter

#         labels = [self.emotion2idx[label] for label in ['anger', 'joy', 'surprise']]
#         labels.reverse()
#         lens = [len(df[df['labels'] == label]) for label in labels]

#         total = sum(lens)
#         ratios = [length/total for length in lens]
#         lens = [int(nsamples*ratio) for ratio in ratios]

#         if sum(lens) != nsamples:
#             lens[-1] += nsamples - sum(lens)

#         for rank in range(target_rank_below):
#             for label, m in zip(labels, lens):
#                 cc = df[df['labels'] == label].iloc[:m]
#                 # cc.loc[:, 'labels'] = cc['labels'].apply(lambda x: self.bmap[self.idx2emotion[label]])

#                 df_split[rank] = pd.concat([df_split[rank], cc])
#                 df[df['labels'] == label] = df[df['labels'] == label].iloc[m:]

#         # print(f"total {sum(lens)}: {lens=}")
#         # for rank in range(target_rank_below):
#         #     print(f"{rank}: {len(df_split[rank])}, labels: {[self.idx2emotion[idx] for idx in df_split[rank].labels.unique()]}")

#         current_rank = target_rank_below
#         for emotion, rank_shift in zip(['anger', 'joy', 'surprise'], [2,5,2]):
#             labels = [self.emotion2idx[label] for label in self.mapping[emotion] if label not in ['anger', 'joy', 'surprise']]
#             labels.reverse()
#             lens = [len(df[df['labels'] == label]) for label in labels]

#             total = sum(lens)
#             ratios = [length/total for length in lens]
#             lens = [int(nsamples*ratio) for ratio in ratios]

#             if sum(lens) != nsamples:
#                 lens[-1] += nsamples - sum(lens)

#             for rank in range(current_rank, current_rank+rank_shift):
#                 for label, m in zip(labels, lens):
#                     cc = df[df['labels'] == label].iloc[:m]
#                     # cc.loc[:, 'labels'] = cc['labels'].apply(lambda x: self.bmap[emotion])
#                     df_split[rank] = pd.concat([df_split[rank], cc])
#                     df[df['labels'] == label] = df[df['labels'] == label].iloc[m:]

#             # print(f"total {sum(lens)}: {lens=}")
#             # for rank in range(current_rank, current_rank+rank_shift):
#             #     print(f"{rank}: {len(df_split[rank])}, labels: {[self.idx2emotion[idx] for idx in df_split[rank].labels.unique()]}")

#             current_rank += rank_shift

#         for emotion in ['sadness', 'fear', 'disgust']:
#             labels = [self.emotion2idx[label] for label in self.mapping[emotion]]

#             for label in labels:
#                 cc = df[df['labels'] == label].iloc[:nsamples]
#                 # cc.loc[:, 'labels'] = cc['labels'].apply(lambda x: self.bmap[emotion])

#                 df_split[current_rank] = cc

#                 n = len(df_split[current_rank])

#                 label = self.emotion2idx['neutral']
#                 cc = df[df['labels'] == label].iloc[:nsamples-n]
#                 # cc.loc[:, 'labels'] = cc['labels'].apply(lambda x: self.bmap['neutral'])
#                 df_split[current_rank] = pd.concat([df_split[current_rank], cc])
#                 df[df['labels'] == label] = df[df['labels'] == label].iloc[nsamples-n:]
#                 current_rank += 1

#         # for rank in range(11, current_rank):
#         #     print(f"{rank}: {len(df_split[rank])}, labels: {[self.idx2emotion[idx] for idx in df_split[rank].labels.unique()]}")

#         label = self.emotion2idx['neutral']
#         for rank in range(current_rank, npeers):
#             cc = df[df['labels'] == label].iloc[:nsamples]
#             # cc.loc[:, 'labels'] = cc['labels'].apply(lambda x: self.bmap['neutral'])
#             df_split[rank] = cc
#             df[df['labels'] == label] = df[df['labels'] == label].iloc[nsamples:]

#         for rank in range(npeers):
#             print(f"{rank}: {len(df_split[rank])}, labels: {[idx for idx in df_split[rank].labels.unique()]}")
#             # print(f"{rank}: {len(df_split[rank])}, labels: {[self.idx2emotion[idx] for idx in df_split[rank].labels.unique()]}")

#         train_split = list()
#         for train_df in df_split:
#             train_input_ids, train_att_masks = self.encode(train_df['text'].values.tolist())
#             train_y = torch.LongTensor(train_df['labels'].values.tolist())
#             train_split.append(TensorDataset(train_input_ids, train_att_masks, train_y))

#         return train_split, trueweights, (None, len(self.classes))

#     def split_test(self, mdnsamples):
#         if self.cfg.valenabled is False:
#             raise RuntimeError(f"Config error: {self.cfg.valenabled=} is not implemented")

#         # split = defaultdict(list)
#         split = list()
#         for key in ['validation', 'test']:
#             df = self.dset[key][:]
#             mask = df['labels'].apply(lambda x: len(x)==1)
#             df = df.loc[mask]
#             df.loc[:, 'labels'] = df['labels'].apply(lambda x: x[0])

#             dfs = list()
#             for label in ['anger', 'joy', 'surprise']:
#                 cc = df[df['labels'] == self.emotion2idx[label]]
#                 # cc.loc[:, 'labels'] = cc['labels'].apply(lambda x: self.bmap[label])
#                 dfs.append(cc)

#             split.append(pd.concat(dfs))

#         valtest = list()
#         for df in split:
#             # print(f"{df=}")
#             # train_input_ids, train_att_masks = self.encode(df['text'])
#             input_ids, att_masks = self.encode(df['text'].values.tolist())
#             # f['text'].map(tokenize_function, batched=True)
#             y = torch.LongTensor(df['labels'].values.tolist())
#             valtest.append(TensorDataset(input_ids, att_masks, y))

#         # print(f"test: {len(split[1])}, labels: {[self.idx2emotion[idx] for idx in split[1].labels.unique()]}")
#         # print(f"val: {len(split[0])}, labels: {[self.idx2emotion[idx] for idx in split[0].labels.unique()]}")
#         print(f"test: {len(split[1])}, labels: {[idx for idx in split[1].labels.unique()]}")
#         print(f"val: {len(split[0])}, labels: {[idx for idx in split[0].labels.unique()]}")
#         return valtest[0], valtest[1]

    
    #############################################################################################################################################################################################
    def split_train(self):
        nclasses = self.cfg.nclasses
        if nclasses != 28:
            raise RuntimeError(f"nclasses != 28")
        npeers = self.cfg.npeers
        if npeers != 20:
            raise RuntimeError(f"npeers != 20")

        hratio = self.cfg.hratio

        df = self.dset['train'][:]
        mask = df['labels'].apply(lambda x: len(x)==1)
        df = df.loc[mask]
        df.loc[:, 'labels'] = df['labels'].apply(lambda x: x[0])

        target_rank_below, near_target_rank_below = 1, 11
        trueweights = [1 / target_rank_below if i < target_rank_below else 0 for i in range(npeers)]

        print(f"{trueweights=}")
        df_split = [pd.DataFrame() for _ in range(npeers)]

        nsamples = 1000
        if nsamples > 1250:
            raise RuntimeError(f"nsamples>1250")

        if hratio is None:
            hratio = 0.99  # no matter

        labels = [self.emotion2idx[label] for label in self.mapping['joy']]
        # labels.reverse()
        lens = [len(df[df['labels'] == label]) for label in labels]

        total = sum(lens)
        ratios = [length/total for length in lens]

        for rank in range(near_target_rank_below):
            lbls = labels.copy()

            if rank < target_rank_below:
                lens = [int(nsamples*ratio) for ratio in ratios]
                if sum(lens) != nsamples:
                    lens[-1] += nsamples - sum(lens)
            else:
                lens = [int(nsamples*ratio*hratio) for ratio in ratios]
                lens.append(nsamples - sum(lens))
                # lbls = [label for label in labels]#.copy()
                lbls.append(self.emotion2idx['neutral'])

            for label, m in zip(lbls, lens):
                cc = df[df['labels'] == label].iloc[:m]
                # cc.loc[:, 'labels'] = cc['labels'].apply(lambda x: self.bmap[self.idx2emotion[label]])

                df_split[rank] = pd.concat([df_split[rank], cc])
                df[df['labels'] == label] = df[df['labels'] == label].iloc[m:]

#             print(f"{rank}: labels: {[self.idx2emotion[idx] for idx in df_split[rank].labels.unique()]}")
#             print(f"{rank}:   lens: {lens}")
#             print(f"{rank}: a lens: {[len(df_split[rank][df_split[rank]['labels'] == idx]) for idx in df_split[rank].labels.unique()]}")

#             print(f"{rank}: {len(df_split[rank])}=")
#             print(f"")

        for emotion, n in zip(["anger", "disgust", "fear", "sadness", "surprise"], [3, 1, 1, 2, 2]):

            labels = [self.emotion2idx[label] for label in self.mapping[emotion]]
            # labels.reverse()

            for k in range(n):
                lbls = labels.copy()

                rank += 1
                lens = [len(df[df['labels'] == label]) for label in labels]

                total = sum(lens)
                ratios = [length/total for length in lens]

                lens = [min(length, int(nsamples*length/total)) for length in lens]
                lens.append(nsamples - sum(lens))
                lbls.append(self.emotion2idx['neutral'])

                for label, m in zip(lbls, lens):
                    cc = df[df['labels'] == label].iloc[:m]
                    # cc.loc[:, 'labels'] = cc['labels'].apply(lambda x: self.bmap[self.idx2emotion[label]])

                    df_split[rank] = pd.concat([df_split[rank], cc])
                    df[df['labels'] == label] = df[df['labels'] == label].iloc[m:]

#                 print(f"{rank}: labels: {[self.idx2emotion[idx] for idx in df_split[rank].labels.unique()]}")
#                 print(f"{rank}:   lens: {lens}")
#                 print(f"{rank}: a lens: {[len(df_split[rank][df_split[rank]['labels'] == idx]) for idx in df_split[rank].labels.unique()]}")

#                 print(f"{rank}: {len(df_split[rank])}=")
#                 print(f"")

        train_split = list()
        for train_df in df_split:
            train_input_ids, train_att_masks = self.encode(train_df['text'].values.tolist())
            train_y = torch.LongTensor(train_df['labels'].values.tolist())
            train_split.append(TensorDataset(train_input_ids, train_att_masks, train_y))

        return train_split, trueweights, (None, len(self.classes))

    def split_test(self, mdnsamples):
        # if self.cfg.valenabled is False:
        #     raise RuntimeError(f"Config error: {self.cfg.valenabled=} is not implemented")

        # split = defaultdict(list)
        split = list()
        for key in ['validation', 'test']:
            df = self.dset[key][:]
            mask = df['labels'].apply(lambda x: len(x)==1)
            df = df.loc[mask]
            df.loc[:, 'labels'] = df['labels'].apply(lambda x: x[0])

            dfs = list()

            for label in self.mapping['joy']:
                cc = df[df['labels'] == self.emotion2idx[label]]
                # cc.loc[:, 'labels'] = cc['labels']
                # cc.loc[:, 'labels'] = cc['labels'].apply(lambda x: self.bmap[label])
                dfs.append(cc)

            split.append(pd.concat(dfs))

        valtest = list()
        for df in split:
            # print(f"{df=}")
            # train_input_ids, train_att_masks = self.encode(df['text'])
            input_ids, att_masks = self.encode(df['text'].values.tolist())
            # f['text'].map(tokenize_function, batched=True)
            y = torch.LongTensor(df['labels'].values.tolist())
            valtest.append(TensorDataset(input_ids, att_masks, y))

        # print(f"test: {len(split[1])}, labels: {[self.idx2emotion[idx] for idx in split[1].labels.unique()]}")
        # print(f"val: {len(split[0])}, labels: {[self.idx2emotion[idx] for idx in split[0].labels.unique()]}")
        # print(f"test: {len(split[1])}, labels: {[idx for idx in split[1].labels.unique()]}")
        # print(f"val: {len(split[0])}, labels: {[idx for idx in split[0].labels.unique()]}")
        if self.cfg.valenabled_ is False:
            return None, valtest[1]
        return valtest[0], valtest[1]
