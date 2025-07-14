from data.preprocess import Preprocess
from data.basic_dataset import RawDataset, BasicDataset
from typing import List
from utils._utils import file_utils
import glob
import torch
import flwr as fl
import numpy as np
import os

# get vocab from multiple datasets
def get_all_vocab(dirs: List[str]) -> List[str]:
    '''
        x = get_all_vocab(["../../data/20NG", "../../data/IMDB"]) -> List words as global vocab
    '''
    all_vocab_set = set()
    for dir in dirs:
        vocab = file_utils.read_text(f'{dir}/vocab.txt')
        all_vocab_set.update(word for word in vocab)  # Sử dụng update thay vì add
    result = list(all_vocab_set)
    result.sort()
    return result

# split data from 1 datasets
def split_data(dir:str, num_split:int, vocab = None, batch_size = 200, device = "cuda") -> List[RawDataset]:
    '''
        datasets = split_data("../../data/20NG", 2, vocab = x)
        -> list RawDataset from a BasicDataset
    '''
    dataset = BasicDataset(dir, batch_size=batch_size, device = device)

    train_texts = dataset.train_texts
    if vocab is None:
        vocab = dataset.vocab
        
    num_sample = int(len(train_texts) / num_split)

    datasets = []
    for i in range(num_split):
        dataset = RawDataset(train_texts[(i * num_sample) : ((i + 1) * num_sample)], vocab = vocab, device = device)
        datasets.append(dataset)
    
    return datasets

def get_latest_server_model(net, max_round):
    # list_of_files = [fname for fname in glob.glob("model_parameters/model_round_*")]
    latest_round_file = f"model_parameters/model_round_{max_round}.npz"
    # print(list_of_files)
    print("Loading pre-trained model from: ", latest_round_file)
    
    # Load NumPy arrays from .npz file
    with np.load(latest_round_file) as data:
        arrays = [data[f'arr_{i}'] for i in range(len(data.files))]
    
    # Convert to PyTorch state_dict
    state_dict = {k: torch.from_numpy(v) for k, v in zip(net.state_dict().keys(), arrays)}
    net.load_state_dict(state_dict)
    
    # Convert to Flower Parameters
    state_dict_ndarrays = [v.cpu().numpy() for v in net.state_dict().values()]
    parameters = fl.common.ndarrays_to_parameters(state_dict_ndarrays)
    return parameters