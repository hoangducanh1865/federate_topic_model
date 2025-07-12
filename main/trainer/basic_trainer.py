import sys
project_root = "d:/MachineLearning/federated_vae"
sys.path.append(project_root)

import torch
from main.data.basic_dataset import BasicDataset
import torch.nn as nn
from collections import defaultdict
import numpy as np
from main.utils import _utils

class BasicTrainer:
    def __init__(self, 
                 model : nn.Module,
                 dataset : BasicDataset,
                 num_top_words = 15,
                 epochs = 200,
                 learning_rate = 0.002,
                 batch_size = 200,
                 verbose = False,
                 log_interval = 1,
                 device = "cuda"):
        self.model = model
        self.dataset = dataset
        self.num_top_words = num_top_words
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.verbose = verbose
        self.log_interval = log_interval
        self.data_size = len(self.dataset.train_data)
        self.device = device

    def make_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)
    
    def train(self, model_name = None):
        optimizer = self.make_optimizer()


        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            for batch_data in self.dataset.train_dataloader:
                batch_data = batch_data.to(self.device)
                output = self.model(batch_data)

                batch_loss = output['loss']

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                total_loss += batch_loss * len(batch_data)

            if (epoch % self.log_interval == 0):
                if model_name is not None:
                    print(f"Client's model: {model_name}")    
                print(f"Epoch: {epoch:03d} | Loss: {total_loss / self.data_size}")

        # top_words = self.get_top_words()
        # train_theta = self.test(self.dataset.train_data)

        # return top_words, train_theta
        return total_loss / self.data_size

    def test(self, bow):
        data_size = bow.shape[0]
        theta = list()
        all_idx = torch.split(torch.arange(data_size), self.batch_size)
        with torch.no_grad():
            self.model.eval()
            for idx in all_idx:
                batch_input = bow[idx]
                batch_input = batch_input.to(self.device)
                # print(batch_input.device)
                batch_theta = self.model.get_theta(batch_input)
                theta.extend(batch_theta.cpu().tolist())

        theta = np.asarray(theta)
        return theta

    def get_beta(self):
        beta = self.model.get_beta().detach().cpu().numpy()
        return beta

    def get_top_words(self, num_top_words=None):
        if num_top_words is None:
            num_top_words = self.num_top_words
        beta = self.get_beta()
        top_words = _utils.get_top_words(beta, self.dataset.vocab, num_top_words, self.verbose)
        return top_words

    def export_theta(self):
        train_theta = self.test(self.dataset.train_data)
        test_theta = self.test(self.dataset.test_data)
        return train_theta, test_theta

