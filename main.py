# https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
import random
import torch.nn.functional as F
import torch.nn as nn
from io import open
import glob
import os
import time
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

import string


def letterToIndex(letter):
    return allowed_characters.find(letter)

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors


def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


class NamesDataset(Dataset):

    def __init__(self, data_dir):
        self.data_dir = data_dir  # for provenance of the dataset
        self.load_time = time.localtime  # for provenance of the dataset
        labels_set = set()  # set of all classes

        self.data = []
        self.data_tensors = []
        self.labels = []
        self.labels_tensors = []

        # read all the ``.txt`` files in the specified directory
        text_files = glob.glob(os.path.join(data_dir, '*.txt'))
        for filename in text_files:
            label = os.path.splitext(os.path.basename(filename))[0]
            labels_set.add(label)
            lines = open(filename, encoding='utf-8').read().strip().split('\n')
            for name in lines:
                self.data.append(name)
                self.data_tensors.append(lineToTensor(name))
                self.labels.append(label)

        # Cache the tensor representation of the labels
        self.labels_uniq = list(labels_set)
        for idx in range(len(self.labels)):
            temp_tensor = torch.tensor(
                [self.labels_uniq.index(self.labels[idx])], dtype=torch.long)
            self.labels_tensors.append(temp_tensor)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        data_label = self.labels[idx]
        data_tensor = self.data_tensors[idx]
        label_tensor = self.labels_tensors[idx]

        return label_tensor, data_tensor, data_label, data_item


class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharRNN, self).__init__()

        self.rnn = nn.RNN(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, line_tensor):
        rnn_out, hidden = self.rnn(line_tensor)
        output = self.h2o(hidden[0])
        output = self.softmax(output)

        return output


def label_from_output(output, output_labels):
    top_n, top_i = output.topk(1)
    label_i = top_i[0].item()
    return output_labels[label_i], label_i


def train(rnn, training_data, n_epoch=10, n_batch_size=64, report_every=50, learning_rate=0.2, criterion=nn.NLLLoss()):
    """
    Learn on a batch of training_data for a specified number of iterations and reporting thresholds
    """
    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []
    rnn.train()
    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

    start = time.time()
    print(f"training on data set with n = {len(training_data)}")

    for iter in range(1, n_epoch + 1):
        rnn.zero_grad()  # clear the gradients

        # create some minibatches
        # we cannot use dataloaders because each of our names is a different length
        batches = list(range(len(training_data)))
        random.shuffle(batches)
        batches = np.array_split(batches, len(batches) // n_batch_size)

        for idx, batch in tqdm(enumerate(batches), total=len(batches)):
            batch_loss = 0
            for i in batch:  # for each example in this batch
                (label_tensor, text_tensor, label, text) = training_data[i]
                output = rnn.forward(text_tensor)
                loss = criterion(output, label_tensor)
                batch_loss += loss

            # optimize parameters
            batch_loss.backward()
            nn.utils.clip_grad_norm_(rnn.parameters(), 3)
            optimizer.step()
            optimizer.zero_grad()

            current_loss += batch_loss.item() / len(batch)

        all_losses.append(current_loss / len(batches))
        if iter % report_every == 0:
            print(
                f"{iter} ({iter / n_epoch:.0%}): \t average batch loss = {all_losses[-1]}")
        current_loss = 0

    return all_losses


if __name__ == "__main__":

    allowed_characters = string.ascii_letters + " .,;'"
    n_letters = len(allowed_characters)
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    alldata = NamesDataset("data/names")
    print(f"loaded {len(alldata)} items of data")
    print(f"example = {alldata[0]}")

    train_set, test_set = torch.utils.data.random_split(
        alldata, [.85, .15], generator=torch.Generator(device=device).manual_seed(2024))

    print(
        f"train examples = {len(train_set)}, validation examples = {len(test_set)}")
    n_hidden = 128
    rnn = CharRNN(n_letters, n_hidden, len(alldata.labels_uniq))
    # print(rnn)
    # input = lineToTensor('Albert')
    # this is equivalent to ``output = rnn.forward(input)``
    # output = rnn(input)
    # print(output)
    # print(label_from_output(output, alldata.labels_uniq))

    start = time.time()
    all_losses = train(rnn, train_set, n_epoch=27,
                       learning_rate=0.15, report_every=5)
    end = time.time()
    print(f"training took {end-start}s")

    # plt.figure()
    # plt.plot(all_losses)
    # plt.show()
