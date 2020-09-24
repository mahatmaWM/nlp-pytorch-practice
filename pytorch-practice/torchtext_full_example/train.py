#!/usr/bin/env python
# -*- coding: utf-8 -*-

from model import *
from data import *
import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
# Hyperparameters

input_size = len(X.vocab)
hidden_size = 512
num_layers = 2
embedding_size = 100
learning_rate = 0.005
num_epochs = 10

# Initialize network
model = RNN_LSTM(input_size, embedding_size, hidden_size, num_layers).to(device)

# (NOT COVERED IN YOUTUBE VIDEO): Load the pretrained embeddings onto our model
pretrained_embeddings = X.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    for batch_idx, batch in tqdm.tqdm(enumerate(train_iter)):
        # Get data to cuda if possible
        data = batch.x.to(device=device)
        targets = batch.y.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores.squeeze(1), targets.type_as(scores))

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent
        optimizer.step()