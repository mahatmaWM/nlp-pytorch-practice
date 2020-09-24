#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ================================================================== #
# 数据集来自 https://github.com/mrthlinh/toxic-comment-classification/tree/master/data
# 根据文本对目标进行二分类，多标签分类
# https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/more_advanced/torchtext
# STEPS:
# 1. Specify how preprocessing should be done -> Fields
# 2. Use Dataset to load the data -> TabularDataset (JSON/CSV/TSV Files)
# 3. Construct an iterator to do batching & padding -> BucketIterator
#
# 1、使用TabularDataset，得到train valid test 3个数据集，并处理embedding向量
# 2、数据集转化为batch迭代器
# 3、使用BatchWrapper将迭代器做封装，转化为tensor数据（根据实际情况选择）
# ================================================================== #

import torch
from torchtext.data import Field, TabularDataset, BucketIterator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenize = lambda x: x.split()
X = Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True)
Y = Field(sequential=False, use_vocab=False)

fields = {"clean_comment": ("x", X), "toxic": ("y", Y)}
# fields = {"clean_comment": ("x", X), "toxic": ("y1", Y), "severe_toxic": ("y2", Y)}

train_data, test_data = TabularDataset.splits(path='./toxic',
                                              train='data_train_clean.csv',
                                              test='data_test_clean.csv',
                                              format='csv',
                                              fields=fields)

X.build_vocab(train_data, max_size=10000, min_freq=1, vectors="glove.6B.100d")

train_iter, test_iter = BucketIterator.splits((train_data, test_data), batch_size=2, device=device)

# batch = next(iter(train_iter))
# print(batch.x)
# print(batch.y1)
# print(batch.y2)
# print([X.vocab.itos[item[1]] for item in batch.x])

class BatchWrapper:

    def __init__(self, dl, x_var, y_vars):
        # 传入自变量x列表和因变量y列表
        self.dl, self.x_var, self.y_vars = dl, x_var, y_vars

    def __iter__(self):
        for batch in self.dl:
            # 在这个封装中只有一个自变量
            x = getattr(batch, self.x_var)
            # 把所有因变量cat成一个向量
            if self.y_vars is not None:
                y = torch.cat([getattr(batch, feat).unsqueeze(1) for feat in self.y_vars], dim=1).float()
            else:
                y = torch.zeros((1))
            yield (x, y)

    def __len__(self):
        return len(self.dl)


# train_dl = BatchWrapper(train_iter, "x", ["y1", "y2"])
# test_dl = BatchWrapper(test_iter, "x", None)

# x, y = next(iter(train_dl))
# print(x)
# print(y)

# for batch in train_iterator:
#     print(batch)

# pretrained_embeddings = X.vocab.vectors
# model.embedding.weight.data.copy_(pretrained_embeddings)